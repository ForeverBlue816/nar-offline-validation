#!/usr/bin/env python3
"""E7 value-cache analysis and E8 one-shot range-direct refinement."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

try:
    from . import experiment as base
    from . import extended_experiment as ext
    from . import activation_experiments as act
except ImportError:
    import experiment as base
    import extended_experiment as ext
    import activation_experiments as act


LOG = logging.getLogger("nar")
MODEL = "llama32_3b"
GROUP_SIZE = 128
VALUE_GROUP_SIZES = (32, 64, 128)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _bf16_memmap(path: Path, mode: str, shape: tuple[int, ...]) -> np.memmap:
    if mode == "r":
        ext._validate_dump_file(path, shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, mode=mode, dtype=np.uint16, shape=shape)


class ValueCapture:
    def __init__(self, model: torch.nn.Module, output: Path, sequences: int, seq_len: int,
                 stride: int, resume: int, moments: list[torch.Tensor], counts: list[int]):
        self.model = model
        self.output = output
        self.sequences = sequences
        self.seq_len = seq_len
        self.stride = stride
        self.positions = list(range(0, seq_len, stride))
        self.layers = int(model.config.num_hidden_layers)
        self.heads = int(model.config.num_key_value_heads)
        self.head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
        self.shape = (sequences, self.heads, len(self.positions), self.head_dim)
        mode = "r+" if resume else "w+"
        self.maps = [
            _bf16_memmap(output / "dumps" / f"layer_{layer:02d}.bf16", mode, self.shape)
            for layer in range(self.layers)
        ]
        self.moments = moments
        self.counts = counts
        self.start = -1
        self.original_attention = model.config._attn_implementation

    def attention(self, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, attention_mask: torch.Tensor | None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        layer = int(module.layer_idx)
        flat = value.detach().float().reshape(-1, self.head_dim)
        self.moments[layer] += (flat.T @ flat).double().cpu()
        self.counts[layer] += flat.shape[0]
        sampled = value[:, :, :: self.stride, :]
        stop = self.start + sampled.shape[0]
        self.maps[layer][self.start:stop] = ext._bf16_bits(sampled)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    def install(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("nar_e7_capture", self.attention)
        self.model.config._attn_implementation = "nar_e7_capture"

    def flush(self) -> None:
        for mmap in self.maps:
            mmap.flush()

    def close(self) -> None:
        self.model.config._attn_implementation = self.original_attention
        self.flush()
        self.maps.clear()


def collect_v(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E7 capture requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e7-collect-v")
    model_id, model_key = act.model_id_and_key(MODEL)
    output = workdir / "activations" / model_key / "v_cal_a"
    done = output / "DONE.json"
    progress = output / "IN_PROGRESS.pt"
    if done.exists():
        LOG.info("E7 capture checkpoint exists: %s", done)
        return
    frozen = {
        "model": model_key, "model_id": model_id, "split": "train", "offset": 0,
        "sequences": args.sequences, "sequence_length": args.seq_len,
        "sample_stride": args.sample_stride, "batch_size": args.batch_size,
    }
    if progress.exists():
        state = torch.load(progress, map_location="cpu", weights_only=True)
        if state["frozen"] != frozen:
            raise RuntimeError("E7 resume settings changed")
        resume = int(state["next_sequence"])
        moments = state["moments"]
        counts = [int(x) for x in state["counts"]]
    else:
        resume = 0
        moments = []
        counts = []
    tokens = base.prepare_token_chunks(model_id, "train", 0, args.sequences, args.seq_len, workdir)
    model = base.load_model(model_id, workdir)
    if not moments:
        head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
        moments = [torch.zeros((head_dim, head_dim), dtype=torch.float64) for _ in range(model.config.num_hidden_layers)]
        counts = [0 for _ in range(model.config.num_hidden_layers)]
    collector = ValueCapture(model, output, args.sequences, args.seq_len, args.sample_stride, resume, moments, counts)
    collector.install()
    started = time.time()
    try:
        with torch.inference_mode():
            for start in range(resume, tokens.shape[0], args.batch_size):
                stop = min(start + args.batch_size, tokens.shape[0])
                collector.start = start
                model.model(input_ids=tokens[start:stop].cuda(non_blocking=True), use_cache=False)
                if stop % args.checkpoint_sequences == 0 or stop == tokens.shape[0]:
                    collector.flush()
                    base.atomic_torch_save(progress, {
                        "frozen": frozen, "next_sequence": stop,
                        "moments": collector.moments, "counts": collector.counts,
                    })
                    LOG.info("E7 capture %d/%d sequences", stop, tokens.shape[0])
    finally:
        collector.close()
    base.atomic_torch_save(output / "moments.pt", {"moments": moments, "counts": counts})
    for layer in range(collector.layers):
        ext._validate_dump_file(output / "dumps" / f"layer_{layer:02d}.bf16", collector.shape)
    base.atomic_json(done, {
        **frozen, "num_layers": collector.layers, "num_key_value_heads": collector.heads,
        "head_dim": collector.head_dim, "sample_positions": collector.positions,
        "sample_shape_per_layer": list(collector.shape),
        "moments": "uncentered second moment uses every token and KV head",
        "samples": "exact bf16 bit patterns at fixed positions for range/NMSE",
        "elapsed_seconds": time.time() - started, "hardware": base.hardware_info(),
    })
    progress.unlink(missing_ok=True)
    del model, collector
    gc.collect()
    torch.cuda.empty_cache()


def _value_rows(output: Path, meta: dict[str, Any], layer: int, device: torch.device) -> torch.Tensor:
    shape = tuple(int(x) for x in meta["sample_shape_per_layer"])
    mmap = _bf16_memmap(output / "dumps" / f"layer_{layer:02d}.bf16", "r", shape)
    return ext._bits_to_tensor(mmap, device).reshape(-1, shape[-1])


def _fold_invariance(r: torch.Tensor, heads: int, seed: int) -> tuple[float, float]:
    n = r.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    y = torch.randn((4, heads, n), generator=generator, dtype=torch.float64)
    weight = torch.randn((min(64, heads * n), heads * n), generator=generator, dtype=torch.float64) / math.sqrt(heads * n)
    rotated = y @ r.T
    folded = (weight.reshape(weight.shape[0], heads, n) @ r.T).reshape(weight.shape[0], heads * n)
    reference = y.reshape(4, -1) @ weight.T
    actual = rotated.reshape(4, -1) @ folded.T
    difference = actual - reference
    return float(difference.abs().max()), float(difference.norm() / reference.norm())


def analyze_e7(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E7 analysis requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e7-value-cache")
    model_id, model_key = act.model_id_and_key(MODEL)
    output = workdir / "activations" / model_key / "v_cal_a"
    meta = _json(output / "DONE.json")
    moments = torch.load(output / "moments.pt", map_location="cpu", weights_only=True)
    result_dir = workdir / "results" / model_key
    done = result_dir / "E7_DONE.json"
    if done.exists():
        LOG.info("E7 checkpoint exists: %s", done)
        return
    device = torch.device("cuda")
    main_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    layer_count = int(meta["num_layers"]) if args.max_layers is None else min(int(meta["num_layers"]), args.max_layers)
    for layer in range(layer_count):
        x = _value_rows(output, meta, layer, device)
        cov = moments["moments"][layer] / int(moments["counts"][layer])
        evals = torch.linalg.eigvalsh((cov + cov.T) / 2).flip(0).clamp_min(0)
        trace = float(evals.sum())
        for b in VALUE_GROUP_SIZES:
            seed = args.seed + 1000 * layer + b
            had = base.full_hadamard_rotation(x.shape[-1], seed).float().to(device)
            raw_range, identity_nmse, _ = base.quant_metrics(x, b)
            had_range, had_nmse, _ = base.quant_metrics(x @ had.T, b)
            final_range = math.nan
            final_nmse = math.nan
            for k in range(x.shape[-1] // b + 1):
                rotation, details = base.nar_rotation(cov, b, seed, absorbed=k)
                rotation = rotation.float().to(device)
                transformed = x @ rotation.T
                range_value, nmse, _ = base.quant_metrics(transformed, b)
                f = min(1.0, max(0.0, float(evals[:k].sum()) / trace))
                rank_rows.append({
                    "model": model_key, "layer": layer, "b": b, "k": k,
                    "dc_slots": x.shape[-1] // b, "mean_group_range": range_value,
                    "relative_quantization_error_nmse": nmse,
                    "range_reduction_vs_hadamard": (had_range - range_value) / had_range,
                    "nmse_delta_vs_hadamard": nmse - had_nmse,
                    "absorbed_energy_fraction": f,
                    "sqrt_one_minus_absorbed_energy_fraction": math.sqrt(1 - f),
                    "orthogonality_max_abs_error": details["orthogonality_max_abs"],
                })
                final_range, final_nmse = range_value, nmse
                if k == x.shape[-1] // b:
                    max_abs, relative = _fold_invariance(rotation.double().cpu(), int(meta["num_key_value_heads"]), seed)
                    invariant_rows.append({
                        "model": model_key, "layer": layer, "b": b,
                        "max_abs_error": max_abs, "relative_l2_error": relative,
                    })
                del rotation, transformed
            for method, range_value, nmse in (
                ("bf16", raw_range, 0.0), ("identity", raw_range, identity_nmse),
                ("hadamard", had_range, had_nmse), ("nar", final_range, final_nmse),
            ):
                main_rows.append({
                    "model": model_key, "layer": layer, "b": b, "method": method,
                    "mean_group_range": range_value, "relative_quantization_error_nmse": nmse,
                    "range_reduction_vs_hadamard": (had_range - range_value) / had_range,
                    "nmse_delta_vs_hadamard": nmse - had_nmse,
                    "sample_vectors": x.shape[0],
                })
        LOG.info("E7 layer %d/%d", layer + 1, meta["num_layers"])
        del x, cov, evals
        torch.cuda.empty_cache()
    fit_rows: list[dict[str, Any]] = []
    for b in VALUE_GROUP_SIZES:
        subset = [r for r in rank_rows if r["b"] == b]
        predictor = np.asarray([r["sqrt_one_minus_absorbed_energy_fraction"] for r in subset])
        # Normalize within each layer against k=0.
        k0 = {r["layer"]: r["mean_group_range"] for r in subset if r["k"] == 0}
        response = np.asarray([r["mean_group_range"] / k0[r["layer"]] for r in subset])
        design = np.column_stack((np.ones_like(predictor), predictor))
        intercept, slope = np.linalg.lstsq(design, response, rcond=None)[0]
        predicted = design @ np.asarray([intercept, slope])
        residual = float(np.square(response - predicted).sum())
        total = float(np.square(response - response.mean()).sum())
        fit_rows.append({
            "model": model_key, "b": b, "intercept": float(intercept), "slope": float(slope),
            "r_squared": 1 - residual / total, "rmse": math.sqrt(residual / len(response)),
            "points": len(response),
            "fit": "OLS range(k)/range(k=0) = intercept + slope*sqrt(1-f), pooled layer-k",
        })
    summary_rows: list[dict[str, Any]] = []
    for b in VALUE_GROUP_SIZES:
        for method in ("bf16", "identity", "hadamard", "nar"):
            subset = [r for r in main_rows if r["b"] == b and r["method"] == method]
            summary_rows.append({
                "model": model_key, "b": b, "method": method, "layers": len(subset),
                "mean_group_range": float(np.mean([r["mean_group_range"] for r in subset])),
                "mean_relative_quantization_error_nmse": float(np.mean([r["relative_quantization_error_nmse"] for r in subset])),
                "mean_range_reduction_vs_hadamard": float(np.mean([r["range_reduction_vs_hadamard"] for r in subset])),
                "mean_nmse_delta_vs_hadamard": float(np.mean([r["nmse_delta_vs_hadamard"] for r in subset])),
            })
    base.write_csv(result_dir / "e7_per_layer.csv", main_rows)
    base.write_csv(result_dir / "e7_range_vs_k.csv", rank_rows)
    base.write_csv(result_dir / "e7_energy_fit.csv", fit_rows)
    base.write_csv(result_dir / "e7_summary.csv", summary_rows)
    base.write_csv(result_dir / "e7_o_proj_fold_invariance.csv", invariant_rows)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "bits": 4,
        "group_sizes": list(VALUE_GROUP_SIZES), "seed": args.seed,
        "rotation": "per-head V rotation; R^T folded blockwise into o_proj input columns",
        "paired_source": str(output / "dumps"), "capture": meta,
        "hardware": base.hardware_info(),
    })


class DownHeldoutCapture:
    def __init__(self, model: torch.nn.Module, output: Path, sequences: int,
                 seq_len: int, stride: int, resume: int):
        self.model = model
        self.output = output
        self.positions = list(range(0, seq_len, stride))
        self.stride = stride
        self.layers = int(model.config.num_hidden_layers)
        self.n = int(model.config.intermediate_size)
        self.shape = (sequences, len(self.positions), self.n)
        mode = "r+" if resume else "w+"
        self.maps = [
            _bf16_memmap(output / "dumps" / f"layer_{layer:02d}.bf16", mode, self.shape)
            for layer in range(self.layers)
        ]
        self.start = -1
        self.handles: list[Any] = []

    def hook(self, layer: int) -> Callable[..., None]:
        def capture(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            sampled = inputs[0][:, :: self.stride, :]
            stop = self.start + sampled.shape[0]
            self.maps[layer][self.start:stop] = ext._bf16_bits(sampled)
        return capture

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.hook(layer)))

    def flush(self) -> None:
        for mmap in self.maps:
            mmap.flush()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.flush()
        self.maps.clear()


def collect_down_heldout(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E8 held-out capture requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e8-collect-down-heldout")
    model_id, model_key = act.model_id_and_key(MODEL)
    output = workdir / "activations" / model_key / "down_cal_b"
    done = output / "DONE.json"
    progress = output / "IN_PROGRESS.json"
    frozen = {
        "model": model_key, "model_id": model_id, "split": "train",
        "offset": args.offset, "sequences": args.sequences,
        "sequence_length": args.seq_len, "sample_stride": args.sample_stride,
        "batch_size": args.batch_size,
    }
    if done.exists():
        LOG.info("E8 held-out capture checkpoint exists: %s", done)
        return
    resume = 0
    if progress.exists():
        state = _json(progress)
        if state["frozen"] != frozen:
            raise RuntimeError("E8 held-out resume settings changed")
        resume = int(state["next_sequence"])
    tokens = base.prepare_token_chunks(model_id, "train", args.offset, args.sequences, args.seq_len, workdir)
    model = base.load_model(model_id, workdir)
    writer = DownHeldoutCapture(model, output, args.sequences, args.seq_len, args.sample_stride, resume)
    writer.install()
    started = time.time()
    try:
        with torch.inference_mode():
            for start in range(resume, tokens.shape[0], args.batch_size):
                stop = min(start + args.batch_size, tokens.shape[0])
                writer.start = start
                model.model(input_ids=tokens[start:stop].cuda(non_blocking=True), use_cache=False)
                if stop % args.checkpoint_sequences == 0 or stop == tokens.shape[0]:
                    writer.flush()
                    base.atomic_json(progress, {"frozen": frozen, "next_sequence": stop})
                    LOG.info("E8 held-out capture %d/%d sequences", stop, tokens.shape[0])
    finally:
        writer.close()
    for layer in range(writer.layers):
        ext._validate_dump_file(output / "dumps" / f"layer_{layer:02d}.bf16", writer.shape)
    base.atomic_json(done, {
        **frozen, "num_layers": writer.layers, "intermediate_size": writer.n,
        "sample_positions": writer.positions, "sample_shape_per_layer": list(writer.shape),
        "sample_rule": "disjoint calibration-B sequences; exact bf16 at positions 0,32,...,2016",
        "elapsed_seconds": time.time() - started, "hardware": base.hardware_info(),
    })
    progress.unlink(missing_ok=True)
    del model, writer
    gc.collect()
    torch.cuda.empty_cache()


def differentiable_reflectors(vectors: torch.Tensor, b: int) -> list[torch.Tensor]:
    work = vectors
    reflectors: list[torch.Tensor] = []
    for index in range(vectors.shape[1]):
        target = torch.zeros(vectors.shape[0], device=vectors.device, dtype=vectors.dtype)
        target[index * b] = 1.0
        delta = work[:, index] - target
        reflector = delta / delta.norm().clamp_min(1e-12)
        reflectors.append(reflector)
        work = work - 2 * reflector[:, None] * (reflector @ work)[None, :]
    return reflectors


def apply_reflector_list(x: torch.Tensor, reflectors: Iterable[torch.Tensor]) -> torch.Tensor:
    output = x
    for reflector in reflectors:
        output = output - 2 * (output @ reflector).unsqueeze(-1) * reflector
    return output


def differentiable_nar(x: torch.Tensor, vectors: torch.Tensor, source: torch.Tensor,
                       target: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    rows = apply_reflector_list(x.float(), differentiable_reflectors(vectors, GROUP_SIZE))
    permuted = torch.empty_like(rows)
    permuted[:, target] = rows[:, source]
    signed = (permuted * signs).reshape(-1, rows.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    return ext._fast_walsh_hadamard(signed).reshape_as(rows)


def range_surrogate(x: torch.Tensor, p: int) -> torch.Tensor:
    groups = x.reshape(-1, x.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    centered = groups - groups.mean(-1, keepdim=True)
    scale = x.square().mean().sqrt().detach().clamp_min(1e-6)
    normalized = centered.abs() / scale
    return normalized.pow(p).mean(-1).clamp_min(1e-20).pow(1 / p).mean()


def _quant_metrics_batched(x: torch.Tensor, factor: act.RotationFactor,
                           signs: torch.Tensor, batch: int) -> tuple[float, float]:
    totals: dict[str, float] = {}
    for start in range(0, x.shape[0], batch):
        rotated = factor.apply(x[start : start + batch], signs)
        ext._merge_sums(totals, ext._quant_sums(rotated, GROUP_SIZE))
    return totals["range_sum"] / totals["group_count"], totals["error_sum"] / totals["energy_sum"]


def run_e8(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E8 requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e8-range-direct")
    model_id, model_key = act.model_id_and_key(MODEL)
    result_dir = workdir / "results" / model_key
    done = result_dir / "E8_DONE.json"
    if done.exists():
        LOG.info("E8 checkpoint exists: %s", done)
        return
    wide = workdir / "activations" / model_key / "wide_cal_a"
    wide_meta = _json(wide / "DONE.json")
    heldout_dir = workdir / "activations" / model_key / "down_cal_b"
    heldout_meta = _json(heldout_dir / "DONE.json")
    factor_root = act.factor_dir(workdir, model_key)
    eig_root = wide / "analysis" / "eigenspaces"
    partial = result_dir / "e8_per_layer.partial.csv"
    rows: list[dict[str, Any]] = list(base.read_csv(partial)) if partial.exists() else []
    completed = {int(r["layer"]) for r in rows if r["split"] == "heldout_cal_b" and r["method"] == "range_direct"}
    refined_dir = wide / "analysis" / "range_direct_p8"
    refined_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    layer_count = int(wide_meta["num_layers"]) if args.max_layers is None else min(int(wide_meta["num_layers"]), args.max_layers)
    for layer in range(layer_count):
        if layer in completed:
            continue
        mmap_cal = ext._open_site(wide, wide_meta, "down_input", layer)
        x_cal = ext._sample_site_tokens(mmap_cal, args.sample_stride, device)
        hold_shape = tuple(int(x) for x in heldout_meta["sample_shape_per_layer"])
        mmap_hold = _bf16_memmap(heldout_dir / "dumps" / f"layer_{layer:02d}.bf16", "r", hold_shape)
        x_hold = ext._bits_to_tensor(mmap_hold, device).reshape(-1, hold_shape[-1])
        eig = torch.load(eig_root / f"down_input_layer_{layer:02d}.pt", map_location="cpu", weights_only=True)
        vectors = eig["vectors"].float().to(device)
        initial_factor = act.RotationFactor.load(factor_root / f"down_layer_{layer:02d}.pt", device)
        signs_generator = torch.Generator(device="cpu").manual_seed(act._seed(args.seed, 0, layer, "down"))
        signs = torch.randint(0, 2, (x_cal.shape[1],), generator=signs_generator, dtype=torch.int64).float().mul_(2).sub_(1).to(device)
        indices = torch.randint(0, x_cal.shape[0], (args.steps, args.batch_size), generator=generator)
        initial_objective = float(range_surrogate(
            differentiable_nar(x_cal[indices[0].to(device)], vectors,
                               initial_factor.source_order, initial_factor.target_order, signs), args.p
        ).detach())
        current = vectors.detach().clone()
        losses: list[float] = []
        for step in range(args.steps):
            current.requires_grad_(True)
            batch = x_cal[indices[step].to(device)]
            transformed = differentiable_nar(batch, current, initial_factor.source_order,
                                             initial_factor.target_order, signs)
            loss = range_surrogate(transformed, args.p)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite E8 loss layer={layer} step={step}")
            gradient = torch.autograd.grad(loss, current)[0]
            tangent = gradient - current @ (current.T @ gradient)
            tangent = tangent / tangent.norm().clamp_min(1e-12)
            with torch.no_grad():
                proposal = current - args.learning_rate * tangent
                updated = torch.linalg.qr(proposal, mode="reduced").Q
                alignment = torch.sign(torch.diag(updated.T @ current)).clamp(min=-1, max=1)
                alignment[alignment == 0] = 1
                current = updated * alignment
            losses.append(float(loss.detach()))
            if (step + 1) % 25 == 0:
                LOG.info("E8 layer=%d step=%d/%d objective=%.6f", layer, step + 1, args.steps, losses[-1])
        refs, active, anchor_error = act.reflectors_from_vectors(current, GROUP_SIZE)
        refined_factor = act.RotationFactor(
            n=current.shape[0], b=GROUP_SIZE, reflectors=refs, active=active,
            source_order=initial_factor.source_order, target_order=initial_factor.target_order,
            anchor_error=anchor_error,
        )
        final_objective = float(range_surrogate(
            differentiable_nar(x_cal[indices[0].to(device)], current,
                               initial_factor.source_order, initial_factor.target_order, signs), args.p
        ).detach())
        for split, values in (("calibration_a", x_cal), ("heldout_cal_b", x_hold)):
            initial_range, initial_nmse = _quant_metrics_batched(values, initial_factor, signs, args.metric_batch)
            refined_range, refined_nmse = _quant_metrics_batched(values, refined_factor, signs, args.metric_batch)
            rows.extend((
                {
                    "model": model_key, "layer": layer, "split": split, "method": "second_moment",
                    "mean_group_range": initial_range, "relative_quantization_error_nmse": initial_nmse,
                    "range_delta_vs_second_moment": 0.0, "nmse_delta_vs_second_moment": 0.0,
                },
                {
                    "model": model_key, "layer": layer, "split": split, "method": "range_direct",
                    "mean_group_range": refined_range, "relative_quantization_error_nmse": refined_nmse,
                    "range_delta_vs_second_moment": refined_range - initial_range,
                    "nmse_delta_vs_second_moment": refined_nmse - initial_nmse,
                },
            ))
        base.write_csv(partial, rows)
        base.atomic_torch_save(refined_dir / f"layer_{layer:02d}.pt", {
            "vectors": current.cpu(), "reflectors": refs.cpu(), "active": active.cpu(),
            "source_order": initial_factor.source_order.cpu(), "target_order": initial_factor.target_order.cpu(),
            "initial_minibatch_objective": initial_objective,
            "final_minibatch_objective": final_objective,
            "step_losses": losses, "anchor_error": anchor_error,
        })
        LOG.info("E8 complete layer=%d/%d", layer + 1, wide_meta["num_layers"])
        del mmap_cal, mmap_hold, x_cal, x_hold, eig, vectors, current, initial_factor, refined_factor
        gc.collect()
        torch.cuda.empty_cache()
    numeric_rows: list[dict[str, Any]] = []
    for row in rows:
        numeric_rows.append({key: (float(value) if key not in ("model", "split", "method") else value)
                             for key, value in row.items()})
    summary: list[dict[str, Any]] = []
    for split in ("calibration_a", "heldout_cal_b"):
        for method in ("second_moment", "range_direct"):
            subset = [r for r in numeric_rows if r["split"] == split and r["method"] == method]
            summary.append({
                "model": model_key, "split": split, "method": method, "layers": len(subset),
                "mean_group_range": float(np.mean([r["mean_group_range"] for r in subset])),
                "mean_relative_quantization_error_nmse": float(np.mean([r["relative_quantization_error_nmse"] for r in subset])),
                "mean_range_delta_vs_second_moment": float(np.mean([r["range_delta_vs_second_moment"] for r in subset])),
                "mean_nmse_delta_vs_second_moment": float(np.mean([r["nmse_delta_vs_second_moment"] for r in subset])),
            })
    held_refined = [r for r in numeric_rows if r["split"] == "heldout_cal_b" and r["method"] == "range_direct"]
    held_summary = next(r for r in summary if r["split"] == "heldout_cal_b" and r["method"] == "range_direct")
    decision = {
        "heldout_mean_range_improved": held_summary["mean_range_delta_vs_second_moment"] < 0,
        "heldout_mean_nmse_improved": held_summary["mean_nmse_delta_vs_second_moment"] < 0,
        "layers_range_improved": sum(r["range_delta_vs_second_moment"] < 0 for r in held_refined),
        "layers_nmse_improved": sum(r["nmse_delta_vs_second_moment"] < 0 for r in held_refined),
        "layers": len(held_refined),
    }
    base.write_csv(result_dir / "e8_per_layer.csv", numeric_rows)
    base.write_csv(result_dir / "e8_summary.csv", summary)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "site": "down_proj input",
        "manifold": "Grassmann projected Riemannian gradient with QR retraction",
        "objective": f"group-centered normalized p-norm range surrogate, p={args.p}",
        "steps": args.steps, "learning_rate": args.learning_rate,
        "gradient_normalization": "unit Frobenius tangent step", "batch_size": args.batch_size,
        "seed": args.seed, "group_size": GROUP_SIZE,
        "permutation_and_signs": "frozen from second-moment NAR; only V is refined",
        "heldout": heldout_meta, "decision": decision, "no_tuning": True,
        "hardware": base.hardware_info(),
    })
    partial.unlink(missing_ok=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    sub = parser.add_subparsers(dest="command", required=True)
    value = sub.add_parser("collect-v")
    value.add_argument("--sequences", type=int, default=128)
    value.add_argument("--seq-len", type=int, default=2048)
    value.add_argument("--sample-stride", type=int, default=32)
    value.add_argument("--batch-size", type=int, default=2)
    value.add_argument("--checkpoint-sequences", type=int, default=8)
    e7 = sub.add_parser("e7")
    e7.add_argument("--max-layers", type=int)
    held = sub.add_parser("collect-down-heldout")
    held.add_argument("--offset", type=int, default=128)
    held.add_argument("--sequences", type=int, default=128)
    held.add_argument("--seq-len", type=int, default=2048)
    held.add_argument("--sample-stride", type=int, default=32)
    held.add_argument("--batch-size", type=int, default=2)
    held.add_argument("--checkpoint-sequences", type=int, default=8)
    refine = sub.add_parser("e8")
    refine.add_argument("--steps", type=int, default=200)
    refine.add_argument("--p", type=int, default=8)
    refine.add_argument("--learning-rate", type=float, default=0.05)
    refine.add_argument("--batch-size", type=int, default=128)
    refine.add_argument("--metric-batch", type=int, default=256)
    refine.add_argument("--sample-stride", type=int, default=32)
    refine.add_argument("--max-layers", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    dispatch = {
        "collect-v": collect_v, "e7": analyze_e7,
        "collect-down-heldout": collect_down_heldout, "e8": run_e8,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
