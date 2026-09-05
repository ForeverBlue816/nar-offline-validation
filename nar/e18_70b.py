#!/usr/bin/env python3
"""E18 Llama-3.1-70B activation-only PPL on a sharded 4/8-GPU model.

The formal run is frozen to one seed, 128 WikiText-2 calibration sequences,
and 64 test sequences at context 2048.  Only the post-RMSNorm q/k/v inputs
and down_proj inputs are fake-quantized with dynamic asymmetric group-128
INT4.  Weights and every other tensor remain bf16.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:
    from . import activation_experiments as act
    from . import experiment as base
    from .e12_wy import compact_wy
except ImportError:
    import activation_experiments as act
    import experiment as base
    from e12_wy import compact_wy


LOG = logging.getLogger("nar")
GROUP = 128
MODEL_ID = "unsloth/Meta-Llama-3.1-70B"
MODEL_KEY = "llama31_70b"
METHODS = ("bf16", "hadamard", "nar_k8", "nar_kmax")


def resolve_ranks(requested: tuple[Any, ...], maximum: int) -> list[tuple[str, int]]:
    """Map requested ranks onto one site; "max" is n/GROUP and ints are capped."""
    resolved: list[tuple[str, int]] = []
    for item in requested:
        if item == "max":
            resolved.append(("nar_kmax", maximum))
        else:
            resolved.append((f"nar_k{int(item)}", min(int(item), maximum)))
    return resolved


def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.model.embed_tokens.weight.device


def load_sharded_model(model_id: str, workdir: Path,
                       dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """Shard the model across the visible GPUs.

    ``dtype`` selects the container precision.  bfloat16 is the original E18
    path.  float32 holds the same bf16 values in fp32 containers, which is what
    E19 does on the 8B: it does not change the weights, it stops the forward and
    the rotation round trip from rounding, and at 70B that rounding is the floor
    the exact-transpose control could not get under.
    """
    from transformers import AutoModelForCausalLM

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("E18 requires CUDA")
    total_gib = [torch.cuda.get_device_properties(i).total_memory // 2**30 for i in range(gpu_count)]
    # Leave room for activations, logits, rotations, and temporary weight-fold buffers.
    max_memory = {i: f"{max(8, gib - 10)}GiB" for i, gib in enumerate(total_gib)}
    LOG.info("loading %s on %d GPUs as %s with max_memory=%s",
             model_id, gpu_count, dtype, max_memory)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(workdir / "cache" / "huggingface"),
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        device_map="balanced",
        max_memory=max_memory,
    )
    return model.eval()


def model_pass(model: torch.nn.Module, tokens: torch.Tensor, batch_size: int, label: str) -> None:
    device = _input_device(model)
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], batch_size):
            stop = min(start + batch_size, tokens.shape[0])
            model.model(input_ids=tokens[start:stop].to(device, non_blocking=True), use_cache=False)
            if stop % max(4, batch_size) == 0 or stop == tokens.shape[0]:
                LOG.info("%s %d/%d sequences", label, stop, tokens.shape[0])


def selected_layers(model: torch.nn.Module, maximum: int) -> int:
    available = len(model.model.layers)
    return available if maximum <= 0 else min(maximum, available)


def make_bases(dimensions: dict[str, int], layers: int, oversample: int, seed: int) -> dict[tuple[str, int], torch.Tensor]:
    bases: dict[tuple[str, int], torch.Tensor] = {}
    for site, n in dimensions.items():
        width = n // GROUP + oversample
        for layer in range(layers):
            generator = torch.Generator(device="cpu").manual_seed(seed + 100_000 * (site == "down") + layer)
            omega = torch.randn((n, width), generator=generator, dtype=torch.float64)
            bases[(site, layer)] = torch.linalg.qr(omega, mode="reduced").Q.float()
    return bases


class MultiRankEnergyCollector:
    """Collect coordinate energies after k=8 and k=max compact-WY maps."""

    def __init__(self, model: torch.nn.Module, vectors: dict[tuple[str, int], torch.Tensor],
                 stride: int, ranks: tuple[Any, ...] = (8, "max")):
        self.model = model
        self.stride = stride
        self.ranks = ranks
        self.handles: list[Any] = []
        self.cpu_factors: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}
        self.device_wy: dict[tuple[str, int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.sums: dict[tuple[str, int, int], torch.Tensor] = {}
        self.counts: dict[tuple[str, int, int], int] = {}
        for key, value in vectors.items():
            maximum = value.shape[1]
            for rank in sorted({r for _, r in resolve_ranks(self.ranks, maximum)}):
                refs, active, error = act.reflectors_from_vectors(value[:, :rank], GROUP)
                w, y = compact_wy(refs, active)
                record = (refs.cpu(), active.cpu(), w.cpu(), y.cpu(), error)
                self.cpu_factors[(key[0], key[1], rank)] = record
                self.sums[(key[0], key[1], rank)] = torch.zeros(value.shape[0], dtype=torch.float64)
                self.counts[(key[0], key[1], rank)] = 0

    def consume(self, site: str, layer: int, value: torch.Tensor) -> None:
        sampled = value.detach()[:, :: self.stride, :].float().reshape(-1, value.shape[-1])
        maximum = value.shape[-1] // GROUP
        for rank in sorted({r for _, r in resolve_ranks(self.ranks, maximum)}):
            key = (site, layer, rank)
            cache_key = (*key, str(sampled.device))
            if cache_key not in self.device_wy:
                _, _, w, y, _ = self.cpu_factors[key]
                self.device_wy[cache_key] = (w.to(sampled.device), y.to(sampled.device))
            w, y = self.device_wy[cache_key]
            transformed = torch.addmm(sampled, sampled @ w, y.T, beta=1.0, alpha=-1.0)
            self.sums[key] += transformed.square().sum(0).double().cpu()
            self.counts[key] += transformed.shape[0]

    def q_hook(self, layer: int) -> Callable[..., torch.Tensor]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            self.consume("qkv", layer, output)
            return output
        return hook

    def down_hook(self, layer: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            self.consume("down", layer, inputs[0])
        return hook

    def install(self, layers: int) -> None:
        for layer, block in enumerate(self.model.model.layers[:layers]):
            self.handles.append(block.input_layernorm.register_forward_hook(self.q_hook(layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.down_hook(layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.device_wy.clear()


def factor_root(workdir: Path, model_key: str) -> Path:
    return workdir / "activations" / model_key / "e18_factors"


def calibrate(
    args: argparse.Namespace,
    model: torch.nn.Module,
    model_id: str,
    model_key: str,
    dimensions: dict[str, int],
    layers: int,
    ranks: tuple[Any, ...] = (8, "max"),
    root_override: Path | None = None,
    eigenspace_name: str = "e18_calibration_eigenspace.csv",
) -> None:
    root = root_override or factor_root(Path(args.workdir), model_key)
    done = root / "DONE.json"
    if done.exists():
        LOG.info("reusing E18 factors: %s", done)
        return
    tokens = base.prepare_token_chunks(
        model_id, "train", 0, args.calibration_sequences, args.seq_len, Path(args.workdir)
    )
    bases = make_bases(dimensions, layers, args.oversample, args.seed)
    final_cq: dict[tuple[str, int], torch.Tensor] = {}
    traces: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    for pass_index in range(3):
        collector = act.SketchCollector(model, bases, trace=pass_index == 2)
        collector.install()
        # Remove hooks on unselected smoke-test layers.
        for handle in collector.handles[2 * layers:]:
            handle.remove()
        collector.handles = collector.handles[: 2 * layers]
        try:
            model_pass(model, tokens, args.batch_size, f"70B covariance pass {pass_index + 1}/3")
        finally:
            collector.close()
        for key, result in collector.results.items():
            if key[1] >= layers:
                continue
            normalized = result / collector.counts[key]
            if pass_index < 2:
                bases[key] = torch.linalg.qr(normalized, mode="reduced").Q.float()
            else:
                final_cq[key] = normalized
                traces[key] = collector.traces[key] / collector.counts[key]
                counts[key] = collector.counts[key]
        del collector
        gc.collect()
        torch.cuda.empty_cache()
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    eigenvalues: dict[tuple[str, int], torch.Tensor] = {}
    residuals: dict[tuple[str, int], torch.Tensor] = {}
    for key in sorted(final_cq):
        q = bases[key]
        cq = final_cq[key]
        small = q.double().T @ cq.double()
        small = (small + small.T) / 2
        values, u = torch.linalg.eigh(small)
        maximum = q.shape[0] // GROUP
        order = torch.argsort(values, descending=True)[:maximum]
        values = values[order].clamp_min(0)
        u = u[:, order]
        v = q.double() @ u
        residual = cq.double() @ u - v * values.unsqueeze(0)
        vectors[key] = v.float()
        eigenvalues[key] = values.cpu()
        residuals[key] = (residual.norm(dim=0) / values.clamp_min(torch.finfo(torch.float64).tiny)).cpu()
    energy = MultiRankEnergyCollector(model, vectors, args.permutation_stride, ranks)
    energy.install(layers)
    try:
        model_pass(model, tokens, args.batch_size, "70B permutation-energy pass 1/1")
    finally:
        energy.close()
    eig_rows: list[dict[str, Any]] = []
    anchor_errors: list[float] = []
    for key, v in vectors.items():
        site, layer = key
        maximum = v.shape[1]
        for label, rank in resolve_ranks(ranks, maximum):
            refs, active, _w, _y, error = energy.cpu_factors[(site, layer, rank)]
            means = energy.sums[(site, layer, rank)] / energy.counts[(site, layer, rank)]
            source, target = act.balanced_orders(means.clamp_min(0).sqrt().unsqueeze(0), rank, GROUP)
            factor = act.RotationFactor(v.shape[0], GROUP, refs, active, source, target, error)
            factor.save(root / label / f"{site}_layer_{layer:02d}.pt", {
                "source": "E18 streamed randomized eigenspace", "site": site, "layer": layer,
                "rank": rank, "rows": counts[key], "eigenvalues": eigenvalues[key][:rank],
                "trace": traces[key], "relative_ritz_residuals": residuals[key][:rank],
            })
            anchor_errors.append(error)
        for index, value in enumerate(eigenvalues[key]):
            eig_rows.append({
                "model": model_key, "site": site, "layer": layer, "rank": index + 1,
                "eigenvalue": float(value), "fraction_total_energy": float(value) / traces[key],
                "cumulative_fraction_total_energy": float(eigenvalues[key][: index + 1].sum()) / traces[key],
                "relative_ritz_residual": float(residuals[key][index]),
            })
    base.write_csv(Path(args.workdir) / "results" / model_key / eigenspace_name, eig_rows)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "num_layers": layers,
        "hidden_size": dimensions["qkv"], "intermediate_size": dimensions["down"],
        "group_size": GROUP, "slot_counts": {"qkv": dimensions["qkv"] // GROUP, "down": dimensions["down"] // GROUP},
        "ranks": list(ranks), "calibration_split": "train", "calibration_offset": 0,
        "calibration_sequences": args.calibration_sequences, "sequence_length": args.seq_len,
        "subspace_oversample": args.oversample, "covariance_passes": 3,
        "permutation_energy_passes": 1, "permutation_token_stride": args.permutation_stride,
        "max_anchor_error": max(anchor_errors), "hardware": base.hardware_info(),
    })
    del tokens, bases, final_cq, vectors, energy
    gc.collect()
    torch.cuda.empty_cache()


class ShardedFactor:
    def __init__(self, path: Path, device: torch.device):
        self.factor = act.RotationFactor.load(path, device)
        self.w, self.y = compact_wy(self.factor.reflectors, self.factor.active)

    def apply(self, value: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        rows = value.float().reshape(-1, self.factor.n)
        rows = torch.addmm(rows, rows @ self.w, self.y.T, beta=1.0, alpha=-1.0)
        permuted = torch.empty_like(rows)
        permuted[:, self.factor.target_order] = rows[:, self.factor.source_order]
        blocks = (permuted * signs).reshape(-1, self.factor.n // GROUP, GROUP)
        return act.ext._fast_walsh_hadamard(blocks).reshape(shape)


class ShardedRotations:
    def __init__(self, args: argparse.Namespace, model: torch.nn.Module, model_key: str, method: str, dimensions: dict[str, int], layers: int):
        self.method = method
        self.signs: dict[tuple[str, int], torch.Tensor] = {}
        self.factors: dict[tuple[str, int], ShardedFactor] = {}
        for layer, block in enumerate(model.model.layers[:layers]):
            for site, module in (("qkv", block.self_attn.q_proj), ("down", block.mlp.down_proj)):
                device = _device_of(module)
                generator = torch.Generator(device="cpu").manual_seed(act._seed(args.seed, 0, layer, site))
                signs = torch.randint(0, 2, (dimensions[site],), generator=generator, dtype=torch.int64)
                self.signs[(site, layer)] = signs.float().mul_(2).sub_(1).to(device)
                if method.startswith("nar_"):
                    path = factor_root(Path(args.workdir), model_key) / method / f"{site}_layer_{layer:02d}.pt"
                    self.factors[(site, layer)] = ShardedFactor(path, device)

    def apply(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        signs = self.signs[(site, layer)]
        if self.method == "hadamard":
            return act.full_hadamard_rows(value.float(), signs)
        return self.factors[(site, layer)].apply(value, signs)


class QuantHooks:
    def __init__(self, model: torch.nn.Module, rotations: ShardedRotations, layers: int):
        self.model = model
        self.rotations = rotations
        self.layers = layers
        self.handles: list[Any] = []

    def quantize(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        rotated = self.rotations.apply(site, layer, value)
        dequant, _, _, _ = base.dynamic_asym_int4(rotated, GROUP)
        return dequant.to(value.dtype)

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers[:self.layers]):
            self.handles.append(block.input_layernorm.register_forward_hook(
                lambda _m, _i, output, layer=layer: self.quantize("qkv", layer, output)
            ))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(
                lambda _m, inputs, layer=layer: (self.quantize("down", layer, inputs[0]),)
            ))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ShardedWeights:
    def __init__(self, model: torch.nn.Module, layers: int):
        self.modules: dict[tuple[str, int], list[torch.nn.Linear]] = {}
        self.originals: dict[int, torch.Tensor] = {}
        for layer, block in enumerate(model.model.layers[:layers]):
            qkv = [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]
            down = [block.mlp.down_proj]
            self.modules[("qkv", layer)] = qkv
            self.modules[("down", layer)] = down
            for module in qkv + down:
                self.originals[id(module)] = module.weight.detach().to("cpu", dtype=torch.bfloat16).clone()

    def restore_all(self) -> None:
        with torch.no_grad():
            for modules in self.modules.values():
                for module in modules:
                    module.weight.copy_(self.originals[id(module)].to(module.weight.device))

    def rotate_all(self, rotations: ShardedRotations, row_batch: int) -> float:
        maximum_error = 0.0
        with torch.no_grad():
            for (site, layer), modules in self.modules.items():
                for module in modules:
                    original = self.originals[id(module)]
                    for start in range(0, original.shape[0], row_batch):
                        stop = min(start + row_batch, original.shape[0])
                        chunk = original[start:stop].to(module.weight.device).float()
                        module.weight[start:stop].copy_(rotations.apply(site, layer, chunk).to(torch.bfloat16))
                    generator = torch.Generator(device="cpu").manual_seed(777 + layer + (site == "down") * 1000)
                    probe = torch.randn((2, original.shape[1]), generator=generator).to(module.weight.device)
                    reference = probe @ original[: min(8, original.shape[0])].to(probe.device).float().T
                    folded = rotations.apply(site, layer, probe) @ module.weight[: min(8, module.weight.shape[0])].float().T
                    maximum_error = max(maximum_error, float((folded - reference).norm() / reference.norm().clamp_min(1e-30)))
        return maximum_error


def evaluate(model: torch.nn.Module, tokens: torch.Tensor, label: str) -> list[float]:
    losses: list[float] = []
    input_device = _input_device(model)
    with torch.inference_mode():
        for index in range(tokens.shape[0]):
            batch = tokens[index:index + 1].to(input_device, non_blocking=True)
            logits = model(input_ids=batch, use_cache=False).logits
            labels = batch[:, 1:].to(logits.device)
            # fp32 BEFORE log_softmax: casting the bf16 loss afterwards quantizes
            # every per-chunk NLL to a bf16 grid (~0.008 at NLL 2.5) and makes
            # paired row comparisons meaningless.  See E18 v2 Step 0.
            loss = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
            if loss.dtype != torch.float32:
                raise AssertionError(f"per-chunk NLL must be fp32, got {loss.dtype}")
            value = float(loss.cpu())
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite {label} loss at sequence {index}")
            losses.append(value)
            if index % 4 == 0 or index + 1 == tokens.shape[0]:
                LOG.info("%s sequence %d/%d loss=%.6f", label, index + 1, tokens.shape[0], value)
            del logits, labels, loss
    return losses


def run(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, f"e18-{args.model_key}")
    base.seed_everything(args.seed)
    result_dir = workdir / "results" / args.model_key
    done = result_dir / "E18_DONE.json"
    if done.exists():
        # Idempotence guard. The 4-GPU and 8-GPU launchers are queued together
        # as a race for whichever GPU count frees up first; only one can run at
        # a time under the per-user GPU limit, and the loser must not re-run a
        # completed experiment if it is scheduled later. Same reason the
        # calibration step already skips on its own checkpoint.
        LOG.info("E18 checkpoint exists, nothing to do: %s", done)
        return
    model = load_sharded_model(args.model_id, workdir)
    layers = selected_layers(model, args.max_layers)
    dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
    LOG.info("E18 dimensions=%s layers=%d slots=%s", dimensions, layers, {k: v // GROUP for k, v in dimensions.items()})
    calibrate(args, model, args.model_id, args.model_key, dimensions, layers)
    tokens = base.prepare_token_chunks(args.model_id, "test", 0, args.eval_sequences, args.seq_len, workdir)
    partial = result_dir / "e18_per_sequence.partial.csv"
    rows: list[dict[str, Any]] = base.read_csv(partial) if partial.exists() else []
    completed = {str(row["method"]) for row in rows}
    weights: ShardedWeights | None = None
    fold_errors: dict[str, float] = {}
    for method in METHODS:
        if method in completed:
            LOG.info("reusing completed method %s", method)
            continue
        if method == "bf16":
            error = 0.0
            hooks = None
            rotations = None
        else:
            if weights is None:
                weights = ShardedWeights(model, layers)
            weights.restore_all()
            rotations = ShardedRotations(args, model, args.model_key, method, dimensions, layers)
            error = weights.rotate_all(rotations, args.weight_row_batch)
            hooks = QuantHooks(model, rotations, layers)
            hooks.install()
        try:
            losses = evaluate(model, tokens, f"{args.model_key} {method}")
        finally:
            if hooks is not None:
                hooks.close()
        fold_errors[method] = error
        rows.extend({
            "model": args.model_key, "model_id": args.model_id, "site": "both", "method": method,
            "seed": args.seed, "sequence": index, "nll": loss, "tokens_scored": args.seq_len - 1,
            "weight_fold_max_relative_error": error,
        } for index, loss in enumerate(losses))
        base.write_csv(partial, rows)
        del rotations, hooks
        gc.collect()
        torch.cuda.empty_cache()
    by_method = {method: [float(row["nll"]) for row in rows if row["method"] == method] for method in METHODS}
    summary = []
    bf16_ppl = math.exp(sum(by_method["bf16"]) / len(by_method["bf16"]))
    had_ppl = math.exp(sum(by_method["hadamard"]) / len(by_method["hadamard"]))
    for method in METHODS:
        ppl = math.exp(sum(by_method[method]) / len(by_method[method]))
        summary.append({
            "model": args.model_key, "site": "both", "method": method, "seed": args.seed,
            "ppl": ppl, "ppl_delta_vs_bf16": ppl - bf16_ppl,
            "ppl_delta_vs_hadamard": ppl - had_ppl if method != "bf16" else math.nan,
            "effective_bits_per_value": 16.0 if method == "bf16" else 4.25,
        })
    base.write_csv(result_dir / "e18_per_sequence.csv", rows)
    base.write_csv(result_dir / "e18_summary.csv", summary)
    base.atomic_json(result_dir / "E18_DONE.json", {
        "model": args.model_key, "model_id": args.model_id, "seed": args.seed,
        "num_layers": layers, "hidden_size": dimensions["qkv"], "intermediate_size": dimensions["down"],
        "head_dim": int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)),
        "group_size": GROUP, "slot_counts": {k: v // GROUP for k, v in dimensions.items()},
        "methods": list(METHODS), "eval_split": "test", "eval_sequences": args.eval_sequences,
        "sequence_length": args.seq_len, "activation_quantizer": "dynamic asymmetric per-token group-128 INT4",
        "metadata": "fp16 scale plus fp16 real-valued zero per group = 4 + 32/128 = 4.25 bits/value",
        "sites": "post-RMSNorm q/k/v_proj input and down_proj input only",
        "weights_kv_other_activations": "bf16", "weight_fold_max_relative_error": fold_errors,
        "paired_design": "same frozen WikiText-2 chunks for all four rows; one seed per user instruction",
        "hardware": base.hardware_info(), "gpu_count": torch.cuda.device_count(),
    })
    partial.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--model-id", default=MODEL_ID)
    result.add_argument("--model-key", default=MODEL_KEY)
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--calibration-sequences", type=int, default=128)
    result.add_argument("--eval-sequences", type=int, default=64)
    result.add_argument("--seq-len", type=int, default=2048)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--oversample", type=int, default=16)
    result.add_argument("--permutation-stride", type=int, default=32)
    result.add_argument("--weight-row-batch", type=int, default=256)
    result.add_argument("--max-layers", type=int, default=0)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
