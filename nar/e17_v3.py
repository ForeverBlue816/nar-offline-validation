#!/usr/bin/env python3
"""E17 v3: verify, select and benchmark the k-independent rank-k projection."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton

try:
    from . import activation_experiments as act
    from . import e17_v2 as v2
    from . import experiment as base
    from .fold_signed_permutation import FoldedR4
    from .kernels import r4_fused_v2 as k2
    from .kernels import r4_fused_v3 as k3
except ImportError:
    import activation_experiments as act
    import e17_v2 as v2
    import experiment as base
    from fold_signed_permutation import FoldedR4
    from kernels import r4_fused_v2 as k2
    from kernels import r4_fused_v3 as k3


SPECS = v2.SPECS
REFERENCE_TILE = k3.TileConfig(8, 4, 2)


def reference_pack(folded: FoldedR4, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    reference = folded.apply(rows)
    deq, codes, scales, zeros = v2.reference_quant(reference)
    return {"dequant": deq, "codes": codes, "scales": scales, "zeros": zeros,
            "u": rows.float() @ folded.y_prime_fp32}


def check(outputs: k2.PackedInt4, n: int, ref: dict[str, torch.Tensor],
          tolerance: float) -> dict[str, Any]:
    observed_codes = k3.unpack(outputs.codes, n)
    observed_deq = k3.dequantize(outputs, n)
    row = {
        "code_match_fraction": float((observed_codes == ref["codes"]).float().mean()),
        "scale_max_ulp": v2.ulp_distance(outputs.scales, ref["scales"]),
        "zero_max_ulp": v2.ulp_distance(outputs.zeros, ref["zeros"]),
        "dequant_max_abs": float((observed_deq - ref["dequant"]).abs().max()),
    }
    row["passes"] = v2.verification_passes(row, tolerance)
    return row


# --------------------------------------------------------------- backends ---

def projection_backends(folded: FoldedR4, k: int) -> dict[str, dict[str, Any]]:
    """Every Kernel A candidate: the v2 reduction, cuBLAS, and the tensor-core
    dot, each at one, two or three bf16 terms for Y'.  x is already bf16, so its
    products are exact in fp32 and only the rounding of Y' is new."""
    cublas_terms = (folded.y_prime_bf16, folded.y_prime_lo_bf16, folded.y_prime_third_bf16)
    backends: dict[str, dict[str, Any]] = {
        "v2_triton_fp32": {
            "kind": "callable", "splits_fixed": None,
            "call": lambda x, partial: k2.projection_triton_fp32(
                x, folded.y_prime_t_fp32, partial),
        },
    }
    for terms in (1, 2, 3):
        backends[f"cublas_fp32out_{terms}term"] = {
            "kind": "callable", "splits_fixed": 1,
            "call": (lambda x, partial, t=terms: k3.projection_cublas_fp32_out_split(
                x, cublas_terms[:t], partial)),
        }
        backends[f"triton_dot_{terms}term"] = {
            "kind": "config", "terms": terms, "factor": folded.y_prime_pad_terms_bf16,
        }
    return backends


def run_projection(backend: dict[str, Any], x: torch.Tensor, partial: torch.Tensor,
                   k: int, config: k3.ProjectionConfig | None) -> torch.Tensor:
    if backend["kind"] == "callable":
        return backend["call"](x, partial)
    return k3.launch_projection_dot(x, backend["factor"], partial, k, config,
                                    backend["terms"])


def pipeline(backend: dict[str, Any], config: k3.ProjectionConfig | None, tile: k3.TileConfig,
             x: torch.Tensor, folded: FoldedR4, k: int, outputs: k2.PackedInt4,
             partial: torch.Tensor) -> None:
    u = run_projection(backend, x, partial, k, config)
    k3.launch_nar(x, folded.w_h_t_fp32, outputs, u, k, tile)


# ------------------------------------------------------------ verification ---

def verify_backends(folded: FoldedR4, k: int, n: int, suites: dict[str, torch.Tensor],
                    tolerance: float, model: str) -> tuple[list[dict[str, Any]], dict[str, list]]:
    """Verification B for every (backend, config); only passers are timable."""
    rows: list[dict[str, Any]] = []
    accepted: dict[str, list] = {}
    backends = projection_backends(folded, k)
    references = {label: reference_pack(folded, sample) for label, sample in suites.items()}
    for name, backend in backends.items():
        configs: list[k3.ProjectionConfig | None]
        configs = list(k3.PROJECTION_CONFIGS) if backend["kind"] == "config" else [None]
        passing: list[k3.ProjectionConfig | None] = []
        for config in configs:
            if config is not None and n % config.splits:
                continue
            entries = []
            for label, sample in suites.items():
                tokens = int(sample.shape[0])
                if config is not None and tokens < config.block_t and tokens % config.block_t:
                    pass  # masked; still exercised
                splits = config.splits if config is not None else (
                    backend["splits_fixed"] or k2.split_count(tokens, n))
                partial = k3.allocate_partial(tokens, k, splits, sample.device)
                outputs = k3.allocate_outputs(tokens, n, sample.device)
                try:
                    u = run_projection(backend, sample, partial, k, config)
                    k3.launch_nar(sample, folded.w_h_t_fp32, outputs, u, k, REFERENCE_TILE)
                    torch.cuda.synchronize()
                    entry = check(outputs, n, references[label], tolerance)
                    reduced = k3.reduce_partial(u, k)
                    entry["projection_relative_l2"] = float(
                        (reduced - references[label]["u"]).norm()
                        / references[label]["u"].norm().clamp_min(1e-30))
                except Exception as error:  # noqa: BLE001 - a config that cannot run cannot be selected
                    entry = {"passes": False, "error": str(error)[:200]}
                entry.update({"model": model, "k": k, "backend": name,
                              "config": config.label() if config else "-", "suite": label,
                              "rows": tokens})
                entries.append(entry)
                del partial, outputs
            rows.extend(entries)
            if all(entry["passes"] for entry in entries):
                passing.append(config)
        accepted[name] = passing
        gc.collect()
        torch.cuda.empty_cache()
    return rows, accepted


def verify_tiles(folded: FoldedR4, k: int, n: int, suites: dict[str, torch.Tensor],
                 tolerance: float, model: str, backend_name: str,
                 backend: dict[str, Any], config: k3.ProjectionConfig | None
                 ) -> tuple[list[dict[str, Any]], list[k3.TileConfig], list[k3.TileConfig]]:
    """Kernel B tiles are verified the same way, for NAR and for the baseline."""
    rows: list[dict[str, Any]] = []
    references = {label: reference_pack(folded, sample) for label, sample in suites.items()}
    hadamard_refs = {}
    for label, sample in suites.items():
        reference = act.ext._fast_walsh_hadamard(
            sample.float().reshape(-1, n // k3.GROUP, k3.GROUP)).reshape_as(sample)
        deq, codes, scales, zeros = v2.reference_quant(reference)
        hadamard_refs[label] = {"dequant": deq, "codes": codes, "scales": scales, "zeros": zeros}
    nar_ok: list[k3.TileConfig] = []
    had_ok: list[k3.TileConfig] = []
    for tile in k3.TILE_CONFIGS:
        nar_entries, had_entries = [], []
        for label, sample in suites.items():
            tokens = int(sample.shape[0])
            splits = config.splits if config is not None else (
                backend["splits_fixed"] or k2.split_count(tokens, n))
            partial = k3.allocate_partial(tokens, k, splits, sample.device)
            outputs = k3.allocate_outputs(tokens, n, sample.device)
            had_outputs = k3.allocate_outputs(tokens, n, sample.device)
            try:
                u = run_projection(backend, sample, partial, k, config)
                k3.launch_nar(sample, folded.w_h_t_fp32, outputs, u, k, tile)
                torch.cuda.synchronize()
                nar_entry = check(outputs, n, references[label], tolerance)
            except Exception as error:  # noqa: BLE001
                nar_entry = {"passes": False, "error": str(error)[:200]}
            try:
                k3.launch_hadamard(sample, had_outputs, tile)
                torch.cuda.synchronize()
                had_entry = check(had_outputs, n, hadamard_refs[label], tolerance)
            except Exception as error:  # noqa: BLE001
                had_entry = {"passes": False, "error": str(error)[:200]}
            for entry, kind in ((nar_entry, "nar"), (had_entry, "hadamard")):
                entry.update({"model": model, "k": k, "backend": f"kernel_b_{kind}",
                              "config": tile.label(), "suite": label, "rows": tokens})
            nar_entries.append(nar_entry)
            had_entries.append(had_entry)
            rows.extend((nar_entry, had_entry))
            del partial, outputs, had_outputs
        if all(entry["passes"] for entry in nar_entries):
            nar_ok.append(tile)
        if all(entry["passes"] for entry in had_entries):
            had_ok.append(tile)
    return rows, nar_ok, had_ok


# ----------------------------------------------------------------- timing ---

def bench(fn: Callable[[], Any], args: argparse.Namespace) -> float:
    return float(triton.testing.do_bench(fn, warmup=args.warmup, rep=args.repeats))


def benchmark_model(args: argparse.Namespace, workdir: Path, model: str,
                    device: torch.device) -> tuple[list, list, list]:
    spec = SPECS[model]
    n, hidden = int(spec["n"]), int(spec["hidden"])
    generator = torch.Generator(device="cpu").manual_seed(args.seed + n)
    weight = (torch.randn((hidden, n), generator=generator, dtype=torch.float32)
              / math.sqrt(n)).to(device, torch.bfloat16)
    verification: list[dict[str, Any]] = []
    backend_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for k in args.ranks:
        factor = act.RotationFactor.load(v2.factor_path(workdir, model, k), device)
        folded = FoldedR4.from_factor(factor, v2.signs_for(n, 0, args.seed, device))
        suites = {"random": v2.random_rows(args.random_verify_rows, n, args.seed + k + n, device)}
        real = v2.load_real_rows(workdir, model, folded, args.real_verify_rows, device)
        if real is not None:
            suites["real"] = real
        rows, accepted = verify_backends(folded, k, n, suites, args.bf16_tolerance, model)
        verification.extend(rows)
        backends = projection_backends(folded, k)
        usable = {name: cfgs for name, cfgs in accepted.items() if cfgs}
        if not usable:
            raise AssertionError(f"no Kernel A backend passes verification for {model} k={k}")
        seed_name = next(iter(usable))
        tile_rows, nar_tiles, had_tiles = verify_tiles(
            folded, k, n, suites, args.bf16_tolerance, model, seed_name,
            backends[seed_name], usable[seed_name][0])
        verification.extend(tile_rows)
        if not nar_tiles or not had_tiles:
            raise AssertionError(f"no Kernel B tile passes verification for {model} k={k}")
        for name, sample in suites.items():
            del sample
        suites.clear()
        gc.collect()
        torch.cuda.empty_cache()

        for tokens in args.tokens:
            x = v2.random_rows(tokens, n, args.seed + tokens + k + n, device)
            outputs = k3.allocate_outputs(tokens, n, device)
            had_outputs = k3.allocate_outputs(tokens, n, device)

            # Kernel A's channel split sets the width of the partial buffer that
            # Kernel B folds, so the two are not independent: the fastest A
            # config can quadruple B's work. Select the pair on the combined
            # pipeline, which is also the number that gets reported.
            for name, configs in usable.items():
                backend = backends[name]
                for config in configs:
                    splits = config.splits if config is not None else (
                        backend["splits_fixed"] or k2.split_count(tokens, n))
                    partial = k3.allocate_partial(tokens, k, splits, device)
                    run_projection(backend, x, partial, k, config)
                    torch.cuda.synchronize()
                    backend_rows.append({
                        "model": model, "k": k, "tokens": tokens, "backend": name,
                        "config": config.label() if config else "-", "splits": splits,
                        "projection_ms": bench(lambda b=backend, c=config, p=partial:
                                               run_projection(b, x, p, k, c), args),
                    })
                    del partial

            best_pair = None
            for name, configs in usable.items():
                backend = backends[name]
                for config in configs:
                    splits = config.splits if config is not None else (
                        backend["splits_fixed"] or k2.split_count(tokens, n))
                    partial = k3.allocate_partial(tokens, k, splits, device)
                    for tile in nar_tiles:
                        def combined(b=backend, c=config, p=partial, t=tile) -> None:
                            u = run_projection(b, x, p, k, c)
                            k3.launch_nar(x, folded.w_h_t_fp32, outputs, u, k, t)
                        combined()
                        torch.cuda.synchronize()
                        ms = bench(combined, args)
                        if best_pair is None or ms < best_pair[0]:
                            best_pair = (ms, name, config, tile, splits)
                    del partial
            assert best_pair is not None
            nar_ms, backend_name, config, nar_tile, splits = best_pair
            backend = backends[backend_name]
            partial = k3.allocate_partial(tokens, k, splits, device)
            run_projection(backend, x, partial, k, config)
            torch.cuda.synchronize()
            projection_ms = bench(lambda: run_projection(backend, x, partial, k, config), args)
            k3.launch_nar(x, folded.w_h_t_fp32, outputs, partial, k, nar_tile)
            torch.cuda.synchronize()
            kernel_b_ms = bench(lambda: k3.launch_nar(
                x, folded.w_h_t_fp32, outputs, partial, k, nar_tile), args)

            had_times = []
            for tile in had_tiles:
                k3.launch_hadamard(x, had_outputs, tile)
                torch.cuda.synchronize()
                had_times.append((bench(lambda t=tile: k3.launch_hadamard(
                    x, had_outputs, t), args), tile))
            had_ms, had_tile = min(had_times, key=lambda item: item[0])

            def nar_pipeline() -> None:
                u = run_projection(backend, x, partial, k, config)
                k3.launch_nar(x, folded.w_h_t_fp32, outputs, u, k, nar_tile)

            nar_pipeline()
            F.linear(x, weight)
            torch.cuda.synchronize()
            matmul_ms = bench(lambda: F.linear(x, weight), args)
            transform_flops = 4 * n * k + n * int(math.log2(k3.GROUP))
            timing_rows.append({
                "tokens": tokens, "model": model, "k": k,
                "nar_fused_ms": nar_ms, "hadamard_fused_ms": had_ms, "down_matmul_ms": matmul_ms,
                "nar_over_hadamard": nar_ms / had_ms, "nar_over_matmul": nar_ms / matmul_ms,
                "hadamard_over_matmul": had_ms / matmul_ms,
                "kernel_a_ms": projection_ms, "kernel_b_nar_ms": kernel_b_ms,
                "kernel_b_hadamard_ms": had_ms,
                "kernel_a_backend": backend_name,
                "kernel_a_config": config.label() if config else "-",
                "kernel_a_splits": splits,
                "kernel_b_nar_tile": nar_tile.label(), "kernel_b_hadamard_tile": had_tile.label(),
                "transform_flop_over_matmul": transform_flops / (2 * n * hidden),
                "variant": "v3-two-launch-tensorcore",
                "warmup": args.warmup, "repetitions": args.repeats,
            })
            del x, outputs, had_outputs, partial
            gc.collect()
            torch.cuda.empty_cache()
        del factor, folded
        gc.collect()
        torch.cuda.empty_cache()
    return timing_rows, backend_rows, verification


@torch.inference_mode()
def layer_overhead(args: argparse.Namespace, workdir: Path, model: str,
                   timings: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    """Each transform's share of one real decoder layer at 2048 tokens."""
    model_id, _ = act.model_id_and_key(model)
    full = base.load_model(model_id, workdir)
    layer = full.model.layers[0]
    hidden = int(full.config.hidden_size)
    tokens = args.layer_tokens
    states = torch.randn((1, tokens, hidden), device=device, dtype=torch.bfloat16) * 0.02
    position_ids = torch.arange(tokens, device=device).unsqueeze(0)
    position_embeddings = full.model.rotary_emb(states, position_ids)

    def forward() -> None:
        layer(states, attention_mask=None, position_ids=position_ids,
              use_cache=False, position_embeddings=position_embeddings)

    forward()
    torch.cuda.synchronize()
    layer_ms = bench(forward, args)
    del full, states, position_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    rows = []
    for row in timings:
        if row["model"] != model or row["tokens"] != tokens:
            continue
        rows.append({
            "model": model, "k": row["k"], "tokens": tokens,
            "decoder_layer_ms": layer_ms,
            "nar_fused_ms": row["nar_fused_ms"], "hadamard_fused_ms": row["hadamard_fused_ms"],
            "nar_share_of_layer": row["nar_fused_ms"] / (layer_ms + row["nar_fused_ms"]),
            "hadamard_share_of_layer": row["hadamard_fused_ms"] / (layer_ms + row["hadamard_fused_ms"]),
            "nar_over_layer": row["nar_fused_ms"] / layer_ms,
            "hadamard_over_layer": row["hadamard_fused_ms"] / layer_ms,
            "nar_over_matmul": row["nar_over_matmul"],
            "hadamard_over_matmul": row["hadamard_over_matmul"],
        })
    return rows


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E17 v3 requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e17-v3")
    device = torch.device("cuda")
    support = {"torch_mm_out_dtype_fp32": k3.cublas_fp32_out_supported(),
               "torch_version": torch.__version__}
    print(json.dumps(support, indent=2), flush=True)
    all_timings, all_backends, all_verification, all_layers = [], [], [], []
    for model in args.models:
        timings, backends, verification = benchmark_model(args, workdir, model, device)
        result_dir = workdir / "results" / model
        base.write_csv(result_dir / "e17v3_fused_r4_timings.csv", timings)
        base.write_csv(result_dir / "e17v3_kernel_a_backends.csv", backends)
        base.atomic_json(result_dir / "e17v3_verification.json", {
            "model": model, "cublas_support": support,
            "configs_tested": len({(r["backend"], r["config"]) for r in verification}),
            "configs_passed": len({(r["backend"], r["config"]) for r in verification
                                   if all(e["passes"] for e in verification
                                          if (e["backend"], e["config"]) == (r["backend"], r["config"]))}),
            "thresholds": {"code_match_fraction_min": 0.999, "metadata_max_ulp": 1,
                           "dequant_max_abs": args.bf16_tolerance},
            "rows": verification,
        })
        if args.layer_overhead:
            layers = layer_overhead(args, workdir, model, timings, device)
            base.write_csv(result_dir / "e17v3_layer_overhead.csv", layers)
            all_layers.extend(layers)
        print(json.dumps({"timings": timings}, indent=2), flush=True)
        all_timings.extend(timings)
        all_backends.extend(backends)
        all_verification.extend(verification)
    primary = [row for row in all_timings if row["tokens"] == 2048]
    gate = {f"{row['model']}_k{row['k']}": row["nar_over_hadamard"] for row in primary}
    base.atomic_json(workdir / "results" / "llama32_3b" / "E17V3_DONE.json", {
        "models": args.models, "ranks": args.ranks, "tokens": args.tokens,
        "cublas_support": support,
        "primary_gate": "NAR/Hadamard <= 2.0 at 2048 tokens for k=8 and k=32",
        "primary_gate_ratios": gate,
        "primary_gate_passed": bool(gate) and all(value <= 2.0 for value in gate.values()),
        "timings": all_timings, "kernel_a_backends": all_backends,
        "layer_overhead": all_layers,
        "configs_tested": len({(r["model"], r["k"], r["backend"], r["config"])
                               for r in all_verification}),
        "hardware": base.hardware_info(),
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--models", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    result.add_argument("--ranks", nargs="+", type=int, choices=(8, 32), default=[8, 32])
    result.add_argument("--tokens", nargs="+", type=int, default=[1, 32, 2048])
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--random-verify-rows", type=int, default=4)
    result.add_argument("--real-verify-rows", type=int, default=64)
    result.add_argument("--bf16-tolerance", type=float, default=0.03125)
    result.add_argument("--warmup", type=int, default=25)
    result.add_argument("--repeats", type=int, default=100)
    result.add_argument("--layer-tokens", type=int, default=2048)
    result.add_argument("--layer-overhead", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
