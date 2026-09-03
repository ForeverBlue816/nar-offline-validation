#!/usr/bin/env python3
"""Verify and benchmark the E17 v2 offline-folded two-launch R4 kernel."""

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
    from . import experiment as base
    from . import extended_experiment as ext
    from .fold_signed_permutation import FoldedR4
    from .kernels import r4_fused_v2 as kernels
except ImportError:
    import activation_experiments as act
    import experiment as base
    import extended_experiment as ext
    from fold_signed_permutation import FoldedR4
    from kernels import r4_fused_v2 as kernels


SPECS = {
    "llama32_3b": {"label": "Llama-3.2-3B", "n": 8192, "hidden": 3072},
    "llama31_8b": {"label": "Llama-3.1-8B", "n": 14336, "hidden": 4096},
}

# name -> (FoldedR4 attribute holding Y', launcher, deployable).
# cuBLAS fp32 is a numerical control only: casting x to fp32 would add an
# uncounted launch, so it is never selected as the deployed Kernel A.
BACKENDS: dict[str, tuple[str, Callable[..., torch.Tensor], bool]] = {
    "cublas_bf16": ("y_prime_bf16", kernels.projection_cublas, True),
    "triton_fp32": ("y_prime_t_fp32", kernels.projection_triton_fp32, True),
    "cublas_fp32": ("y_prime_fp32", kernels.projection_cublas_fp32, False),
}


def factor_path(workdir: Path, model: str, rank: int, layer: int = 0) -> Path:
    return (
        workdir / "activations" / model / "e11_calibration" / "factors"
        / f"nar_b128_k{rank}" / f"down_layer_{layer:02d}.pt"
    )


def signs_for(n: int, layer: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(act._seed(seed, 0, layer, "down"))
    return torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1).to(device)


def reference_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dequant, scales, zeros, codes = base.dynamic_asym_int4(value, kernels.GROUP)
    return dequant.float(), codes.reshape_as(value), scales, zeros


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    ai = a.detach().cpu().contiguous().view(torch.int16).to(torch.int32)
    bi = b.detach().cpu().contiguous().view(torch.int16).to(torch.int32)
    return int((ai - bi).abs().max())


def projection_inputs(folded: FoldedR4, backend: str, tokens: int, device: torch.device,
                      override: int = 0) -> tuple[torch.Tensor, Callable[..., torch.Tensor], torch.Tensor, int]:
    attribute, launcher, _ = BACKENDS[backend]
    k = folded.w_h_t_fp32.shape[0]
    splits = (kernels.split_count(tokens, folded.n, override=override)
              if backend.startswith("triton") else 1)
    partial = kernels.allocate_partial(tokens, k, splits, device)
    return getattr(folded, attribute), launcher, partial, splits


def verify_rows(label: str, rows: torch.Tensor, folded: FoldedR4,
                backend: str) -> dict[str, Any]:
    n = folded.n
    k = folded.w_h_t_fp32.shape[0]
    outputs = kernels.allocate_outputs(rows.shape[0], n, rows.device)
    y_prime, projection, partial, splits = projection_inputs(
        folded, backend, rows.shape[0], rows.device
    )
    u = kernels.launch_nar(rows, y_prime, folded.w_h_t_fp32, outputs, partial,
                           projection=projection)
    torch.cuda.synchronize()
    u_reduced = kernels.reduce_partial(u, k)
    u_reference = rows.float() @ folded.y_prime_fp32
    reference = folded.apply(rows)
    ref_deq, ref_codes, ref_scales, ref_zeros = reference_quant(reference)
    observed_codes = kernels.unpack(outputs.codes, n)
    observed_deq = kernels.dequantize(outputs, n)
    result = {
        "label": label,
        "rows": int(rows.shape[0]),
        "projection_backend": backend,
        "projection_dtype": str(u.dtype).removeprefix("torch."),
        "projection_splits": splits,
        "projection_max_abs": float((u_reduced - u_reference).abs().max()),
        "projection_relative_l2": float(
            (u_reduced - u_reference).norm() / u_reference.norm().clamp_min(1e-30)
        ),
        "code_match_fraction": float((observed_codes == ref_codes).float().mean()),
        "scale_max_abs": float((outputs.scales - ref_scales).abs().max()),
        "scale_max_ulp": ulp_distance(outputs.scales, ref_scales),
        "zero_max_abs": float((outputs.zeros - ref_zeros).abs().max()),
        "zero_max_ulp": ulp_distance(outputs.zeros, ref_zeros),
        "dequant_max_abs": float((observed_deq - ref_deq).abs().max()),
        "dequant_relative_l2": float((observed_deq - ref_deq).norm() / ref_deq.norm().clamp_min(1e-30)),
    }
    del outputs, u, u_reduced, u_reference, reference, partial
    del ref_deq, ref_codes, ref_scales, ref_zeros, observed_codes, observed_deq
    return result


def verify_fused_rows(label: str, rows: torch.Tensor, folded: FoldedR4) -> dict[str, Any]:
    """Verification B for the Variant 2 single-launch kernel."""
    n = folded.n
    outputs = kernels.allocate_outputs(rows.shape[0], n, rows.device)
    kernels.launch_nar_fused(rows, folded.y_prime_t_fp32, folded.w_h_t_fp32, outputs)
    torch.cuda.synchronize()
    reference = folded.apply(rows)
    ref_deq, ref_codes, ref_scales, ref_zeros = reference_quant(reference)
    observed_codes = kernels.unpack(outputs.codes, n)
    observed_deq = kernels.dequantize(outputs, n)
    result = {
        "label": label,
        "rows": int(rows.shape[0]),
        "projection_backend": "variant2_fused",
        "code_match_fraction": float((observed_codes == ref_codes).float().mean()),
        "scale_max_abs": float((outputs.scales - ref_scales).abs().max()),
        "scale_max_ulp": ulp_distance(outputs.scales, ref_scales),
        "zero_max_abs": float((outputs.zeros - ref_zeros).abs().max()),
        "zero_max_ulp": ulp_distance(outputs.zeros, ref_zeros),
        "dequant_max_abs": float((observed_deq - ref_deq).abs().max()),
        "dequant_relative_l2": float((observed_deq - ref_deq).norm() / ref_deq.norm().clamp_min(1e-30)),
    }
    del outputs, reference, ref_deq, ref_codes, ref_scales, ref_zeros, observed_codes, observed_deq
    return result


def verify_hadamard_rows(label: str, rows: torch.Tensor) -> dict[str, Any]:
    n = rows.shape[1]
    outputs = kernels.allocate_outputs(rows.shape[0], n, rows.device)
    kernels.launch_hadamard(rows, outputs)
    torch.cuda.synchronize()
    reference = act.ext._fast_walsh_hadamard(
        rows.float().reshape(-1, n // kernels.GROUP, kernels.GROUP)
    ).reshape_as(rows)
    ref_deq, ref_codes, ref_scales, ref_zeros = reference_quant(reference)
    observed_codes = kernels.unpack(outputs.codes, n)
    observed_deq = kernels.dequantize(outputs, n)
    result = {
        "label": label,
        "rows": int(rows.shape[0]),
        "code_match_fraction": float((observed_codes == ref_codes).float().mean()),
        "scale_max_abs": float((outputs.scales - ref_scales).abs().max()),
        "scale_max_ulp": ulp_distance(outputs.scales, ref_scales),
        "zero_max_abs": float((outputs.zeros - ref_zeros).abs().max()),
        "zero_max_ulp": ulp_distance(outputs.zeros, ref_zeros),
        "dequant_max_abs": float((observed_deq - ref_deq).abs().max()),
        "dequant_relative_l2": float(
            (observed_deq - ref_deq).norm() / ref_deq.norm().clamp_min(1e-30)
        ),
    }
    del outputs, reference, ref_deq, ref_codes, ref_scales, ref_zeros, observed_codes, observed_deq
    return result


def random_rows(tokens: int, n: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((tokens, n), generator=generator, dtype=torch.float32).to(device, torch.bfloat16)


def real_rows_path(workdir: Path, model: str) -> Path:
    return workdir / "artifacts" / "e17v2" / model / "real_down_rows_layer00.pt"


def load_real_rows(workdir: Path, model: str, folded: FoldedR4,
                   count: int, device: torch.device) -> torch.Tensor | None:
    cached = real_rows_path(workdir, model)
    if cached.exists():
        payload = torch.load(cached, map_location="cpu", weights_only=True)
        saved = payload["down_input"]
        if saved.shape[0] >= count:
            original = saved[:count].to(device, torch.bfloat16)
            return folded.q_unfolded(original).to(torch.bfloat16).contiguous()
        if model != "llama32_3b":
            raise RuntimeError(f"{cached} has {saved.shape[0]} rows; {count} required")
    if model == "llama32_3b":
        wide = workdir / "activations" / model / "wide_cal_a"
        dump = wide / "dumps" / "down_input" / "layer_00.bf16"
        if dump.exists():
            meta = json.loads((wide / "DONE.json").read_text())
            mapped = ext._open_site(wide, meta, "down_input", 0)
            original = ext._bits_to_tensor(mapped.reshape(-1, folded.n)[:count], device).to(torch.bfloat16)
            return folded.q_unfolded(original).to(torch.bfloat16).contiguous()
    return None


def verification_passes(row: dict[str, Any], tolerance: float) -> bool:
    return (
        row["code_match_fraction"] >= 0.999
        and row["scale_max_ulp"] <= 1
        and row["zero_max_ulp"] <= 1
        and row["dequant_max_abs"] <= tolerance
    )


def enforce_verification(rows: list[dict[str, Any]], tolerance: float) -> None:
    print(json.dumps({"verification_rows": rows}, indent=2), flush=True)
    for row in rows:
        if not verification_passes(row, tolerance):
            raise AssertionError(json.dumps(row, indent=2))


def benchmark_model(args: argparse.Namespace, workdir: Path, model: str,
                    device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = SPECS[model]
    n, hidden = int(spec["n"]), int(spec["hidden"])
    generator = torch.Generator(device="cpu").manual_seed(args.seed + n)
    weight = (torch.randn((hidden, n), generator=generator, dtype=torch.float32) / math.sqrt(n)).to(
        device, torch.bfloat16
    )
    timing_rows: list[dict[str, Any]] = []
    verification: list[dict[str, Any]] = []
    had_verified = False
    for rank in args.ranks:
        factor = act.RotationFactor.load(factor_path(workdir, model, rank), device)
        signs = signs_for(n, 0, args.seed, device)
        folded = FoldedR4.from_factor(factor, signs)
        random = random_rows(args.random_verify_rows, n, args.seed + rank + n, device)
        real = load_real_rows(workdir, model, folded, args.real_verify_rows, device)
        candidate_rows: list[dict[str, Any]] = []
        for backend in BACKENDS:
            candidate_rows.append(verify_rows(f"random-k{rank}", random, folded, backend))
            if real is not None:
                candidate_rows.append(verify_rows(f"real-down-layer0-k{rank}", real, folded, backend))
        fused_rows = [verify_fused_rows(f"random-k{rank}", random, folded)]
        if real is not None:
            fused_rows.append(verify_fused_rows(f"real-down-layer0-k{rank}", real, folded))
        candidate_rows.extend(fused_rows)
        for row in candidate_rows:
            row["passes"] = verification_passes(row, args.bf16_tolerance)
        verification.extend(candidate_rows)
        accepted = [
            name for name, (_, _, deployable) in BACKENDS.items()
            if deployable and all(
                row["passes"] for row in candidate_rows
                if row["projection_backend"] == name
            )
        ]
        if not accepted:
            print(json.dumps({"verification_rows": candidate_rows}, indent=2), flush=True)
            raise AssertionError(f"no projection backend passes for {model} k={rank}")
        if not had_verified:
            verification.append(verify_hadamard_rows("random-hadamard", random))
            if real is not None:
                verification.append(verify_hadamard_rows("real-down-layer0-hadamard", real))
            had_verified = True
        del random, real
        gc.collect()
        torch.cuda.empty_cache()

        for tokens in args.tokens:
            x = random_rows(tokens, n, args.seed + tokens + rank + n, device)
            nar_outputs = kernels.allocate_outputs(tokens, n, device)
            had_outputs = kernels.allocate_outputs(tokens, n, device)
            projection_ms: dict[str, float] = {}
            candidates: dict[str, float] = {}
            for name in BACKENDS:
                y_prime, projection, partial, splits = projection_inputs(
                    folded, name, tokens, device, args.projection_splits
                )
                # Compile/autotune outside the timed region.
                kernels.launch_nar(x, y_prime, folded.w_h_t_fp32, nar_outputs, partial,
                                   projection=projection)
                torch.cuda.synchronize()
                projection_ms[name] = float(triton.testing.do_bench(
                    lambda: projection(x, y_prime, partial),
                    warmup=args.warmup, rep=args.repeats,
                ))
                if name in accepted:
                    candidates[name] = float(triton.testing.do_bench(
                        lambda: kernels.launch_nar(
                            x, y_prime, folded.w_h_t_fp32, nar_outputs, partial,
                            projection=projection,
                        ),
                        warmup=args.warmup, rep=args.repeats,
                    ))
                del partial
            backend = min(candidates, key=candidates.get)
            splits = (kernels.split_count(tokens, n, override=args.projection_splits)
                      if backend.startswith("triton") else 1)
            kernels.launch_nar_fused(x, folded.y_prime_t_fp32, folded.w_h_t_fp32, nar_outputs)
            torch.cuda.synchronize()
            fused_ms = float(triton.testing.do_bench(
                lambda: kernels.launch_nar_fused(
                    x, folded.y_prime_t_fp32, folded.w_h_t_fp32, nar_outputs),
                warmup=args.warmup, rep=args.repeats,
            ))

            kernels.launch_hadamard(x, had_outputs)
            F.linear(x, weight)
            torch.cuda.synchronize()
            had_ms = float(triton.testing.do_bench(
                lambda: kernels.launch_hadamard(x, had_outputs),
                warmup=args.warmup, rep=args.repeats,
            ))
            matmul_ms = float(triton.testing.do_bench(
                lambda: F.linear(x, weight), warmup=args.warmup, rep=args.repeats,
            ))
            transform_flops = 4 * n * rank + n * int(math.log2(kernels.GROUP))
            matmul_flops = 2 * n * hidden
            for variant, nar_ms, kernel_a in (
                ("variant-1-two-launch", candidates[backend], backend),
                ("variant-2-single-launch", fused_ms, "fused-in-kernel"),
            ):
                row = {
                    "tokens": tokens,
                    "model": model,
                    "k": rank,
                    "nar_fused_ms": nar_ms,
                    "hadamard_fused_ms": had_ms,
                    "down_matmul_ms": matmul_ms,
                    "nar_over_hadamard": nar_ms / had_ms,
                    "nar_over_matmul": nar_ms / matmul_ms,
                    "hadamard_over_matmul": had_ms / matmul_ms,
                    "transform_flop_over_matmul": transform_flops / matmul_flops,
                    "variant": variant,
                    "kernel_a_backend": kernel_a,
                    "kernel_a_splits": splits if variant.startswith("variant-1") else 0,
                    "warmup": args.warmup,
                    "repetitions": args.repeats,
                }
                row.update({f"{name}_projection_ms": value for name, value in projection_ms.items()})
                row.update({f"{name}_nar_fused_ms": value for name, value in candidates.items()})
                timing_rows.append(row)
            del x, nar_outputs, had_outputs
        del factor, folded, signs
        gc.collect()
        torch.cuda.empty_cache()
    return timing_rows, {
        "model": model,
        "model_label": spec["label"],
        "n": n,
        "hidden": hidden,
        "verification": verification,
        "thresholds": {
            "code_match_fraction_min": 0.999,
            "metadata_max_ulp": 1,
            "dequant_max_abs": args.bf16_tolerance,
        },
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E17 v2 requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e17-v2")
    device = torch.device("cuda")
    all_timings: list[dict[str, Any]] = []
    verifications: dict[str, Any] = {}
    for model in args.models:
        timings, verification = benchmark_model(args, workdir, model, device)
        result_dir = workdir / "results" / model
        base.write_csv(result_dir / "e17v2_fused_r4_timings.csv", timings)
        base.atomic_json(result_dir / "e17v2_verification.json", verification)
        print(json.dumps({"timings": timings}, indent=2), flush=True)
        all_timings.extend(timings)
        verifications[model] = verification
    for model, verification in verifications.items():
        selected = {row["kernel_a_backend"] for row in all_timings if row["model"] == model}
        selected.add("variant2_fused")
        enforce_verification(
            [row for row in verification["verification"]
             if row.get("projection_backend") in (None, *selected)],
            args.bf16_tolerance,
        )
    primary = [row for row in all_timings if row["tokens"] == 2048 and row["k"] == 8]
    best = {row["model"]: min(r["nar_over_hadamard"] for r in primary if r["model"] == row["model"])
            for row in primary}
    passed = bool(primary) and all(value <= 2.0 for value in best.values())
    base.atomic_json(workdir / "results" / "llama32_3b" / "E17V2_DONE.json", {
        "models": args.models,
        "ranks": args.ranks,
        "tokens": args.tokens,
        "timings": all_timings,
        "verifications": verifications,
        "primary_gate": "best-variant NAR/Hadamard <= 2.0 at 2048 tokens for k=8",
        "primary_gate_best_ratio": best,
        "primary_gate_passed": passed,
        "e12_superseded": passed,
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
    result.add_argument("--projection-splits", type=int, default=0,
                        help="fix Kernel A's channel split; 0 uses the occupancy heuristic")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
