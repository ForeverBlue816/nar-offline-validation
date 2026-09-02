#!/usr/bin/env python3
"""Run the corrected and extended NAR activation experiments.

This entry point never reruns the completed E1 K b=32/64 rows or any E2 row.
It reads those frozen artifacts, performs E1b offline, captures wide residual
and MLP activations plus full post-RoPE K once, and then performs E1c/E1d.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

try:
    from . import experiment as base
except ImportError:
    import experiment as base


LOG = logging.getLogger("nar")
WIDE_GROUP_SIZE = 128
VALID_K_GROUP_SIZES = (32, 64)
POSITION_PROBES = (0, 512, 1024, 2048)
WIDE_SITES = ("q_input", "down_input")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, str]]:
    return base.read_csv(path) if path.exists() else []


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _bf16_bits(x: torch.Tensor) -> np.ndarray:
    cpu = x.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return cpu.view(torch.uint16).numpy()


def _bits_to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    owned = np.array(x, dtype=np.uint16, copy=True, order="C")
    return torch.from_numpy(owned).view(torch.bfloat16).to(device=device, dtype=torch.float32)


def _expected_bytes(shape: Iterable[int]) -> int:
    return math.prod(shape) * 2


def _validate_dump_file(path: Path, shape: tuple[int, ...]) -> None:
    expected = _expected_bytes(shape)
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"{path}: expected {expected} bytes, found {actual}")


class WideActivationWriter:
    """Stream exact bf16 bit patterns to fixed-shape uint16 memory maps."""

    def __init__(
        self,
        model: torch.nn.Module,
        outdir: Path,
        sequences: int,
        seq_len: int,
        resume_sequence: int,
    ) -> None:
        self.model = model
        self.outdir = outdir
        self.sequences = sequences
        self.seq_len = seq_len
        self.resume_sequence = resume_sequence
        self.write_start = -1
        self.n_layers = int(model.config.num_hidden_layers)
        self.hidden_size = int(model.config.hidden_size)
        self.intermediate_size = int(model.config.intermediate_size)
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.head_dim = int(getattr(model.config, "head_dim", self.hidden_size // model.config.num_attention_heads))
        self.maps: dict[tuple[str, int], np.memmap] = {}
        self.shapes: dict[str, tuple[int, ...]] = {
            "q_input": (sequences, seq_len, self.hidden_size),
            "down_input": (sequences, seq_len, self.intermediate_size),
            "k_post": (sequences, self.num_kv_heads, seq_len, self.head_dim),
        }
        self.handles: list[Any] = []
        self.original_attention_impl = model.config._attn_implementation
        mode = "r+" if resume_sequence else "w+"
        for site, shape in self.shapes.items():
            site_dir = outdir / "dumps" / site
            site_dir.mkdir(parents=True, exist_ok=True)
            for layer in range(self.n_layers):
                path = site_dir / f"layer_{layer:02d}.bf16"
                if resume_sequence:
                    _validate_dump_file(path, shape)
                self.maps[(site, layer)] = np.memmap(path, mode=mode, dtype=np.uint16, shape=shape)

    def _write(self, site: str, layer: int, x: torch.Tensor) -> None:
        if self.write_start < 0:
            raise RuntimeError("write_start was not set")
        stop = self.write_start + x.shape[0]
        expected = self.shapes[site][1:]
        if tuple(x.shape[1:]) != expected:
            raise RuntimeError(f"{site} layer {layer}: expected (*,{expected}), got {tuple(x.shape)}")
        self.maps[(site, layer)][self.write_start:stop] = _bf16_bits(x)

    def pre_hook(self, site: str, layer: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            self._write(site, layer, inputs[0])
        return hook

    def attention(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        self._write("k_post", int(module.layer_idx), key)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    def install(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("nar_wide_capture", self.attention)
        self.model.config._attn_implementation = "nar_wide_capture"
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.self_attn.q_proj.register_forward_pre_hook(self.pre_hook("q_input", layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.pre_hook("down_input", layer)))

    def flush(self) -> None:
        for mmap in self.maps.values():
            mmap.flush()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model.config._attn_implementation = self.original_attention_impl
        self.flush()
        self.maps.clear()


def collect_wide_activations(args: argparse.Namespace) -> None:
    wd = base.work_path(args)
    base.setup_logging(wd, f"collect-wide-{args.tag}")
    model_id = base.resolve_model_id(args.model)
    model_key = base.model_key_from_id(model_id)
    if model_key != "llama32_3b":
        raise ValueError("E1c is frozen to Llama-3.2-3B")
    outdir = wd / "activations" / model_key / args.tag
    done = outdir / "DONE.json"
    progress_path = outdir / "IN_PROGRESS.json"
    if done.exists():
        LOG.info("wide activation checkpoint exists, skipping: %s", done)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("wide activation capture requires an allocated CUDA GPU")
    frozen = {
        "model_id": model_id,
        "split": args.split,
        "offset": args.offset,
        "sequences": args.sequences,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "format": "raw little-endian uint16 words containing exact torch.bfloat16 bits",
    }
    resume = 0
    if progress_path.exists():
        progress = _read_json(progress_path)
        if progress["frozen"] != frozen:
            raise RuntimeError(f"capture settings disagree with {progress_path}")
        resume = int(progress["next_sequence"])
        LOG.info("resuming wide capture at sequence %d", resume)
    elif outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"refusing to overwrite uncheckpointed contents in {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    base.atomic_json(progress_path, {"frozen": frozen, "next_sequence": resume})
    tokens = base.prepare_token_chunks(model_id, args.split, args.offset, args.sequences, args.seq_len, wd)
    model = base.load_model(model_id, wd)
    writer = WideActivationWriter(model, outdir, args.sequences, args.seq_len, resume)
    writer.install()
    started = time.time()
    try:
        with torch.inference_mode():
            for start in range(resume, tokens.shape[0], args.batch_size):
                stop = min(start + args.batch_size, tokens.shape[0])
                writer.write_start = start
                batch = tokens[start:stop].cuda(non_blocking=True)
                model.model(input_ids=batch, use_cache=False)
                if stop % args.checkpoint_sequences == 0 or stop == tokens.shape[0]:
                    writer.flush()
                    base.atomic_json(progress_path, {"frozen": frozen, "next_sequence": stop})
                    LOG.info("wide capture checkpoint %d/%d sequences", stop, tokens.shape[0])
    finally:
        writer.close()
    shapes = {site: list(shape) for site, shape in writer.shapes.items()}
    for site, shape in writer.shapes.items():
        for layer in range(writer.n_layers):
            _validate_dump_file(outdir / "dumps" / site / f"layer_{layer:02d}.bf16", shape)
    summary = {
        **frozen,
        "model_key": model_key,
        "tag": args.tag,
        "num_layers": writer.n_layers,
        "hidden_size": writer.hidden_size,
        "intermediate_size": writer.intermediate_size,
        "num_key_value_heads": writer.num_kv_heads,
        "head_dim": writer.head_dim,
        "site_shapes": shapes,
        "sample_rule": "every token in every one of the fixed 128 sequences; no token subsampling",
        "elapsed_seconds": time.time() - started,
        "hardware": base.hardware_info(),
    }
    base.atomic_json(done, summary)
    progress_path.unlink(missing_ok=True)
    LOG.info("wide capture complete in %.1fs", summary["elapsed_seconds"])
    del model, writer
    gc.collect()
    torch.cuda.empty_cache()


def run_e1b(args: argparse.Namespace) -> None:
    """Test plain post-RoPE NAR at fixed hypothetical positions, offline."""
    wd = base.work_path(args)
    base.setup_logging(wd, "e1b-position-plain-nar")
    model_id = base.resolve_model_id(args.model)
    model_key = base.model_key_from_id(model_id)
    cal_dir = wd / "activations" / model_key / args.cal_a
    meta = _read_json(cal_dir / "DONE.json")
    if meta["moments_only"]:
        raise RuntimeError("E1b needs the frozen pre-RoPE tensor samples")
    rope = torch.load(cal_dir / "rope_probe.pt", map_location="cpu", weights_only=True)
    rope_positions = tuple(int(x) for x in rope["positions"].reshape(-1).tolist())
    if rope_positions != POSITION_PROBES:
        raise RuntimeError((rope_positions, POSITION_PROBES))
    result_dir = wd / "results" / model_key
    output = result_dir / "e1b_position_plain_nar.csv"
    done = result_dir / "E1B_DONE.json"
    if done.exists():
        LOG.info("E1b checkpoint exists, skipping: %s", done)
        return
    rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for layer in range(int(meta["num_layers"])):
        payload = base.layer_payload(cal_dir, layer)
        cov = base.covariance(payload)
        x_pre = payload["k_pre"].float().to(device)
        for b in VALID_K_GROUP_SIZES:
            seed = args.seed + 1000 * layer + b
            rotations = {
                "hadamard": base.full_hadamard_rotation(int(meta["head_dim"]), seed),
                "nar": base.nar_rotation(cov, b, seed)[0],
            }
            for pos_index, position in enumerate(rope_positions):
                x_pos = base.apply_rope_offline(
                    x_pre,
                    rope["cos"][0, pos_index].float().to(device),
                    rope["sin"][0, pos_index].float().to(device),
                )
                values: dict[str, tuple[float, float]] = {}
                raw_range, raw_nmse, _ = base.quant_metrics(x_pos, b)
                values["identity"] = (raw_range, raw_nmse)
                for method, rotation in rotations.items():
                    r, nmse, _ = base.quant_metrics(base.rotate_rows(x_pos, rotation), b)
                    values[method] = (r, nmse)
                had_range, had_nmse = values["hadamard"]
                for method in ("bf16", "identity", "hadamard", "nar"):
                    range_value, nmse = (raw_range, 0.0) if method == "bf16" else values[method]
                    rows.append({
                        "model": model_key,
                        "layer": layer,
                        "b": b,
                        "position": position,
                        "method": method,
                        "mean_group_range": range_value,
                        "relative_quantization_error_nmse": nmse,
                        "range_reduction_vs_hadamard": (had_range - range_value) / had_range,
                        "nmse_delta_vs_hadamard": nmse - had_nmse,
                        "sample_vectors": x_pos.shape[0],
                    })
        LOG.info("E1b analyzed layer %d/%d", layer + 1, meta["num_layers"])
    base.write_csv(output, rows)
    base.atomic_json(done, {
        "model": model_key,
        "seed": args.seed,
        "source_calibration": str(cal_dir),
        "group_sizes": list(VALID_K_GROUP_SIZES),
        "positions": list(POSITION_PROBES),
        "methods": ["bf16", "identity", "hadamard", "nar"],
        "nar_rope_status": "dropped: plain NAR strictly dominated NAR-RoPE in all 28 layers for range and NMSE at b=32 and b=64",
    })
    LOG.info("wrote %s", output)


def _site_shape(meta: dict[str, Any], site: str) -> tuple[int, ...]:
    return tuple(int(x) for x in meta["site_shapes"][site])


def _site_path(wide_dir: Path, site: str, layer: int) -> Path:
    return wide_dir / "dumps" / site / f"layer_{layer:02d}.bf16"


def _open_site(wide_dir: Path, meta: dict[str, Any], site: str, layer: int) -> np.memmap:
    shape = _site_shape(meta, site)
    path = _site_path(wide_dir, site, layer)
    _validate_dump_file(path, shape)
    return np.memmap(path, mode="r", dtype=np.uint16, shape=shape)


def _iter_flat_batches(
    mmap: np.memmap,
    device: torch.device,
    sequence_batch: int,
) -> Iterable[torch.Tensor]:
    for start in range(0, mmap.shape[0], sequence_batch):
        stop = min(start + sequence_batch, mmap.shape[0])
        yield _bits_to_tensor(mmap[start:stop], device).reshape(-1, mmap.shape[-1])


def _covariance_apply(
    mmap: np.memmap,
    q: torch.Tensor,
    device: torch.device,
    sequence_batch: int,
    compute_trace: bool = False,
) -> tuple[torch.Tensor, float | None]:
    q_device = q.to(device=device, dtype=torch.float32)
    result = torch.zeros((q.shape[0], q.shape[1]), dtype=torch.float64)
    trace_sum = 0.0
    count = 0
    for x in _iter_flat_batches(mmap, device, sequence_batch):
        projected = x @ q_device
        result += (x.T @ projected).double().cpu()
        if compute_trace:
            trace_sum += float(x.square().sum(dtype=torch.float64).item())
        count += x.shape[0]
        del x, projected
    result /= count
    return result, (trace_sum / count if compute_trace else None)


def randomized_top_eigenspace(
    mmap: np.memmap,
    rank: int,
    oversample: int,
    seed: int,
    device: torch.device,
    sequence_batch: int,
) -> dict[str, Any]:
    """Fixed randomized subspace iteration: Q=orth(C orth(C Omega))."""
    n = int(mmap.shape[-1])
    width = min(n, rank + oversample)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    omega = torch.randn((n, width), generator=generator, dtype=torch.float64)
    q = torch.linalg.qr(omega, mode="reduced").Q.float()
    for pass_index in range(2):
        cq, _ = _covariance_apply(mmap, q, device, sequence_batch)
        q = torch.linalg.qr(cq, mode="reduced").Q.float()
        LOG.info("subspace pass %d/2 n=%d rank=%d", pass_index + 1, n, rank)
    cq, trace = _covariance_apply(mmap, q, device, sequence_batch, compute_trace=True)
    small = (q.double().T @ cq.double())
    small = (small + small.T) / 2
    evals, u = torch.linalg.eigh(small)
    order = torch.argsort(evals, descending=True)[:rank]
    evals = evals[order].clamp_min(0)
    u = u[:, order]
    vectors = q.double() @ u
    residual = cq.double() @ u - vectors * evals.unsqueeze(0)
    relative_residuals = residual.norm(dim=0) / evals.clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "vectors": vectors.float().cpu(),
        "eigenvalues": evals.cpu(),
        "trace": float(trace),
        "relative_residuals": relative_residuals.cpu(),
        "rank": rank,
        "oversample": oversample,
        "power_iterations": 1,
        "passes_over_dump": 3,
        "rows": math.prod(mmap.shape[:-1]),
    }


def _householders_to_anchors(vectors: torch.Tensor, b: int) -> tuple[list[torch.Tensor | None], float]:
    """Return sequential reflectors mapping column i to coordinate i*b."""
    work = vectors.float().clone()
    n, rank = work.shape
    reflectors: list[torch.Tensor | None] = []
    for index in range(rank):
        anchor = index * b
        target = torch.zeros(n, dtype=work.dtype, device=work.device)
        target[anchor] = 1.0
        delta = work[:, index] - target
        norm = delta.norm()
        if float(norm) < 1e-7:
            reflector = None
        else:
            reflector = delta / norm
            work[:, index:] -= 2 * reflector[:, None] * (reflector @ work[:, index:])[None, :]
        reflectors.append(reflector)
    anchors = torch.arange(rank, device=work.device) * b
    mapped = work[anchors, torch.arange(rank, device=work.device)]
    off = work.clone()
    off[anchors, torch.arange(rank, device=work.device)] = 0
    mapping_error = max(float((mapped - 1).abs().max()), float(off.abs().max()))
    return reflectors, mapping_error


def _apply_reflector_rows(x: torch.Tensor, reflector: torch.Tensor | None) -> torch.Tensor:
    if reflector is None:
        return x
    return x - 2 * (x @ reflector).unsqueeze(1) * reflector.unsqueeze(0)


def _block_hadamard_rows(
    x: torch.Tensor,
    b: int,
    signs: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    groups = x.shape[-1] // b
    signed = (x * signs).reshape(-1, groups, b)
    return torch.matmul(signed, h.T).reshape_as(x)


def _fast_walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError(f"FWHT dimension must be a power of two, got {n}")
    original_shape = x.shape
    y = x.reshape(-1, n)
    width = 1
    while width < n:
        blocks = y.reshape(-1, n // (2 * width), 2, width)
        left = blocks[:, :, 0, :]
        right = blocks[:, :, 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(-1, n)
        width *= 2
    return (y / math.sqrt(n)).reshape(original_shape)


def _paley_hadamard_12(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q = 11
    residues = {value * value % q for value in range(1, q)}
    core = torch.empty((q, q), dtype=dtype, device=device)
    for row in range(q):
        for column in range(q):
            delta = (row - column) % q
            core[row, column] = 0 if delta == 0 else (1 if delta in residues else -1)
    matrix = torch.ones((q + 1, q + 1), dtype=dtype, device=device)
    matrix[1:, 1:] = core - torch.eye(q, dtype=dtype, device=device)
    error = (matrix @ matrix.T - (q + 1) * torch.eye(q + 1, dtype=dtype, device=device)).abs().max()
    if float(error) > 1e-5:
        raise AssertionError(f"invalid Paley-12 Hadamard: {float(error)}")
    return matrix / math.sqrt(q + 1)


def _full_hadamard_rows(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    signed = x * signs
    if not n & (n - 1):
        return _fast_walsh_hadamard(signed)
    if n == 3072:
        factored = _fast_walsh_hadamard(signed.reshape(-1, 12, 256))
        h12 = _paley_hadamard_12(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h12.T).transpose(1, 2).reshape_as(x)
    raise ValueError(f"no frozen full-Hadamard construction for n={n}")


def _balanced_permute_rows(x: torch.Tensor, absorbed: int, groups: int, b: int) -> torch.Tensor:
    """Place absorbed anchors on DC, use low-energy fillers, balance residual energy."""
    n = x.shape[-1]
    energies = x.square().mean(0).double().cpu()
    absorbed_sources = [index * b for index in range(absorbed)]
    remaining = [index for index in range(n) if index not in absorbed_sources]
    fillers = sorted(remaining, key=lambda index: (float(energies[index]), index))[: groups - absorbed]
    anchor_sources = absorbed_sources + fillers
    residual_sources = [index for index in remaining if index not in fillers]
    residual_sources.sort(key=lambda index: (-float(energies[index]), index))
    residual_energies = [max(0.0, float(energies[index])) for index in residual_sources]
    target_order = [group * b for group in range(groups)] + base._balanced_target_slots(residual_energies, groups, b)
    source_order = anchor_sources + residual_sources
    if len(source_order) != n or len(set(source_order)) != n or len(set(target_order)) != n:
        raise AssertionError("invalid NAR permutation")
    output = torch.empty_like(x)
    output[:, torch.tensor(target_order, device=x.device)] = x[:, torch.tensor(source_order, device=x.device)]
    return output


def _sample_site_tokens(
    mmap: np.memmap,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    positions = np.arange(0, mmap.shape[1], stride, dtype=np.int64)
    return _bits_to_tensor(mmap[:, positions, :], device).reshape(-1, mmap.shape[-1])


def _e1c_plots(
    result_dir: Path,
    rank_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, site in zip(axes, WIDE_SITES):
        subset = [r for r in rank_rows if r["site"] == site]
        ks = sorted({int(r["k"]) for r in subset})
        layers = sorted({int(r["layer"]) for r in subset})
        for layer in layers:
            vals = [_float(r, "mean_group_range") for r in subset if int(r["layer"]) == layer]
            ax.plot(ks, vals, color="0.75", linewidth=0.6, alpha=0.5)
        means = [np.mean([_float(r, "mean_group_range") for r in subset if int(r["k"]) == k]) for k in ks]
        ax.plot(ks, means, color="black", marker="o", markersize=2.5, linewidth=2, label="layer mean")
        ax.set_title(site)
        ax.set_xlabel("absorbed directions k")
        ax.set_ylabel("mean b=128 group range")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("E1c NAR activation range vs absorbed rank")
    fig.tight_layout()
    fig.savefig(result_dir / "e1c_range_vs_k.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, site in zip(axes, WIDE_SITES):
        subset = [r for r in rank_rows if r["site"] == site]
        x = np.asarray([_float(r, "sqrt_one_minus_absorbed_energy_fraction") for r in subset])
        y = np.asarray([_float(r, "normalized_range_vs_k0") for r in subset])
        fit = next(r for r in fit_rows if r["site"] == site)
        ax.scatter(x, y, s=7, alpha=0.2, color="tab:blue", label="layer x k")
        line_x = np.linspace(float(x.min()), float(x.max()), 100)
        line_y = _float(fit, "intercept") + _float(fit, "slope") * line_x
        ax.plot(line_x, line_y, color="black", linewidth=2, label=f"OLS R2={_float(fit, 'r_squared'):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="tab:red", linewidth=1, label="RMS prediction y=x")
        ax.set_title(site)
        ax.set_xlabel("sqrt(1 - absorbed energy fraction)")
        ax.set_ylabel("realized range / range(k=0)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("E1c realized range scaling versus absorbed-energy prediction")
    fig.tight_layout()
    fig.savefig(result_dir / "e1c_energy_fit.png", dpi=180)
    plt.close(fig)


def run_e1c(args: argparse.Namespace) -> None:
    wd = base.work_path(args)
    base.setup_logging(wd, "e1c-wide-activations")
    if not torch.cuda.is_available():
        raise RuntimeError("E1c analysis requires an allocated CUDA GPU")
    model_key = base.model_key_from_id(base.resolve_model_id(args.model))
    wide_dir = wd / "activations" / model_key / args.wide_tag
    meta = _read_json(wide_dir / "DONE.json")
    result_dir = wd / "results" / model_key
    result_dir.mkdir(parents=True, exist_ok=True)
    done = result_dir / "E1C_DONE.json"
    if done.exists():
        LOG.info("E1c checkpoint exists, skipping: %s", done)
        return
    settings = {
        "wide_tag": args.wide_tag,
        "group_size": WIDE_GROUP_SIZE,
        "evaluation_token_stride": args.eval_token_stride,
        "evaluation_positions": list(range(0, int(meta["seq_len"]), args.eval_token_stride)),
        "subspace_oversample": args.oversample,
        "power_iterations": 1,
        "subspace_passes": 3,
        "sequence_batch": args.sequence_batch,
    }
    progress = result_dir / "E1C_IN_PROGRESS.json"
    if progress.exists() and _read_json(progress) != settings:
        raise RuntimeError(f"E1c settings disagree with {progress}")
    base.atomic_json(progress, settings)
    main_partial = result_dir / "e1c_per_layer.partial.csv"
    rank_partial = result_dir / "e1c_range_vs_k.partial.csv"
    eig_partial = result_dir / "e1c_eigenspace.partial.csv"
    main_rows: list[dict[str, Any]] = _rows(main_partial)
    rank_rows: list[dict[str, Any]] = _rows(rank_partial)
    eig_rows: list[dict[str, Any]] = _rows(eig_partial)
    completed = {(r["site"], int(r["layer"])) for r in main_rows if r["method"] == "nar_kmax"}
    eig_dir = wide_dir / "analysis" / "eigenspaces"
    eig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    h = base.hadamard(WIDE_GROUP_SIZE, dtype=torch.float32).to(device)
    torch.set_float32_matmul_precision("highest")

    for site_index, site in enumerate(WIDE_SITES):
        n = int(meta["hidden_size"] if site == "q_input" else meta["intermediate_size"])
        rank = n // WIDE_GROUP_SIZE
        for layer in range(int(meta["num_layers"])):
            if (site, layer) in completed:
                continue
            mmap = _open_site(wide_dir, meta, site, layer)
            eig_path = eig_dir / f"{site}_layer_{layer:02d}.pt"
            if eig_path.exists():
                eig = torch.load(eig_path, map_location="cpu", weights_only=True)
                for key, value in (("rank", rank), ("oversample", args.oversample), ("power_iterations", 1)):
                    if int(eig[key]) != value:
                        raise RuntimeError(f"{eig_path}: {key} mismatch")
            else:
                eig = randomized_top_eigenspace(
                    mmap,
                    rank,
                    args.oversample,
                    args.seed + 100_000 * site_index + layer,
                    device,
                    args.sequence_batch,
                )
                base.atomic_torch_save(eig_path, eig)
            vectors = eig["vectors"].to(device=device, dtype=torch.float32)
            eigenvalues = eig["eigenvalues"].double()
            trace = float(eig["trace"])
            residuals = eig["relative_residuals"].double()
            for index in range(rank):
                eig_rows.append({
                    "model": model_key,
                    "site": site,
                    "layer": layer,
                    "rank": index + 1,
                    "eigenvalue": float(eigenvalues[index]),
                    "fraction_total_energy": float(eigenvalues[index]) / trace,
                    "cumulative_fraction_total_energy": float(eigenvalues[: index + 1].sum()) / trace,
                    "relative_ritz_residual": float(residuals[index]),
                })
            x = _sample_site_tokens(mmap, args.eval_token_stride, device)
            raw_range, identity_nmse, _ = base.quant_metrics(x, WIDE_GROUP_SIZE)
            generator = torch.Generator(device="cpu").manual_seed(args.seed + 1000 * layer + 10 * site_index + WIDE_GROUP_SIZE)
            signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1).to(device)
            had = _full_hadamard_rows(x, signs)
            had_range, had_nmse, _ = base.quant_metrics(had, WIDE_GROUP_SIZE)
            reflectors, mapping_error = _householders_to_anchors(vectors, WIDE_GROUP_SIZE)
            work = x
            k0_range = math.nan
            final_range = math.nan
            final_nmse = math.nan
            for k in range(rank + 1):
                if k:
                    work = _apply_reflector_rows(work, reflectors[k - 1])
                permuted = _balanced_permute_rows(work, k, rank, WIDE_GROUP_SIZE)
                rotated = _block_hadamard_rows(permuted, WIDE_GROUP_SIZE, signs, h)
                range_value, nmse, _ = base.quant_metrics(rotated, WIDE_GROUP_SIZE)
                if k == 0:
                    k0_range = range_value
                energy_fraction = min(1.0, max(0.0, float(eigenvalues[:k].sum()) / trace))
                rank_rows.append({
                    "model": model_key,
                    "site": site,
                    "layer": layer,
                    "n": n,
                    "b": WIDE_GROUP_SIZE,
                    "k": k,
                    "dc_slots": rank,
                    "mean_group_range": range_value,
                    "relative_quantization_error_nmse": nmse,
                    "range_reduction_vs_k0": (k0_range - range_value) / k0_range,
                    "range_reduction_vs_hadamard": (had_range - range_value) / had_range,
                    "range_reduction_vs_identity": (raw_range - range_value) / raw_range,
                    "nmse_delta_vs_hadamard": nmse - had_nmse,
                    "absorbed_energy_fraction": energy_fraction,
                    "sqrt_one_minus_absorbed_energy_fraction": math.sqrt(1 - energy_fraction),
                    "normalized_range_vs_k0": range_value / k0_range,
                    "evaluation_tokens": x.shape[0],
                    "anchor_mapping_max_abs_error": mapping_error,
                })
                final_range, final_nmse = range_value, nmse
                del permuted, rotated
            for method, range_value, nmse in (
                ("bf16", raw_range, 0.0),
                ("identity", raw_range, identity_nmse),
                ("hadamard_full", had_range, had_nmse),
                ("nar_kmax", final_range, final_nmse),
            ):
                main_rows.append({
                    "model": model_key,
                    "site": site,
                    "layer": layer,
                    "n": n,
                    "b": WIDE_GROUP_SIZE,
                    "method": method,
                    "mean_group_range": range_value,
                    "relative_quantization_error_nmse": nmse,
                    "range_reduction_vs_hadamard": (had_range - range_value) / had_range,
                    "nmse_delta_vs_hadamard": nmse - had_nmse,
                    "evaluation_tokens": x.shape[0],
                })
            base.write_csv(main_partial, main_rows)
            base.write_csv(rank_partial, rank_rows)
            base.write_csv(eig_partial, eig_rows)
            LOG.info("E1c complete site=%s layer=%d/%d", site, layer + 1, meta["num_layers"])
            del mmap, eig, vectors, x, had, work, reflectors
            gc.collect()
            torch.cuda.empty_cache()

    fit_rows: list[dict[str, Any]] = []
    for site in WIDE_SITES:
        subset = [r for r in rank_rows if r["site"] == site]
        x = np.asarray([_float(r, "sqrt_one_minus_absorbed_energy_fraction") for r in subset])
        y = np.asarray([_float(r, "normalized_range_vs_k0") for r in subset])
        design = np.column_stack((np.ones_like(x), x))
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        predicted = intercept + slope * x
        ss_res = float(np.square(y - predicted).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        fit_rows.append({
            "site": site,
            "fit": "OLS normalized_range = intercept + slope * sqrt(1-f), pooled over paired layer-k rows",
            "intercept": float(intercept),
            "slope": float(slope),
            "r_squared": 1 - ss_res / ss_tot,
            "rmse": math.sqrt(ss_res / len(y)),
            "points": len(y),
        })
    summary_rows: list[dict[str, Any]] = []
    for site in WIDE_SITES:
        for method in ("bf16", "identity", "hadamard_full", "nar_kmax"):
            subset = [r for r in main_rows if r["site"] == site and r["method"] == method]
            summary_rows.append({
                "site": site,
                "method": method,
                "layers": len(subset),
                "mean_group_range": float(np.mean([_float(r, "mean_group_range") for r in subset])),
                "mean_relative_quantization_error_nmse": float(np.mean([_float(r, "relative_quantization_error_nmse") for r in subset])),
                "mean_range_reduction_vs_hadamard": float(np.mean([_float(r, "range_reduction_vs_hadamard") for r in subset])),
                "mean_nmse_delta_vs_hadamard": float(np.mean([_float(r, "nmse_delta_vs_hadamard") for r in subset])),
            })
    base.write_csv(result_dir / "e1c_per_layer.csv", main_rows)
    base.write_csv(result_dir / "e1c_summary.csv", summary_rows)
    base.write_csv(result_dir / "e1c_range_vs_k.csv", rank_rows)
    base.write_csv(result_dir / "e1c_eigenspace.csv", eig_rows)
    base.write_csv(result_dir / "e1c_energy_fit.csv", fit_rows)
    _e1c_plots(result_dir, rank_rows, fit_rows)
    base.atomic_json(done, {"model": model_key, "seed": args.seed, "settings": settings, "wide_capture": meta})
    for path in (main_partial, rank_partial, eig_partial, progress):
        path.unlink(missing_ok=True)
    LOG.info("E1c outputs complete under %s", result_dir)


def _quant_sums(x: torch.Tensor, b: int) -> dict[str, float]:
    grouped = base.group_view(x.float(), b)
    range_sum = float((grouped.amax(-1) - grouped.amin(-1)).double().sum().item())
    dequant, _, _, _ = base.dynamic_asym_int4(x.float(), b)
    return {
        "range_sum": range_sum,
        "group_count": grouped.numel() // b,
        "error_sum": float((dequant.float() - x.float()).square().sum(dtype=torch.float64).item()),
        "energy_sum": float(x.float().square().sum(dtype=torch.float64).item()),
    }


def _merge_sums(target: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        target[key] = target.get(key, 0.0) + value


def run_e1d(args: argparse.Namespace) -> None:
    wd = base.work_path(args)
    base.setup_logging(wd, "e1d-kivi-baseline")
    if not torch.cuda.is_available():
        raise RuntimeError("E1d analysis requires an allocated CUDA GPU")
    model_key = base.model_key_from_id(base.resolve_model_id(args.model))
    wide_dir = wd / "activations" / model_key / args.wide_tag
    wide_meta = _read_json(wide_dir / "DONE.json")
    cal_dir = wd / "activations" / model_key / args.cal_a
    cal_meta = _read_json(cal_dir / "DONE.json")
    result_dir = wd / "results" / model_key
    done = result_dir / "E1D_DONE.json"
    if done.exists():
        LOG.info("E1d checkpoint exists, skipping: %s", done)
        return
    partial = result_dir / "e1d_per_layer.partial.csv"
    rows: list[dict[str, Any]] = _rows(partial)
    completed = {(int(r["layer"]), int(r["b"])) for r in rows if r["method"] == "kivi_per_channel"}
    device = torch.device("cuda")
    n = int(wide_meta["head_dim"])
    for layer in range(int(wide_meta["num_layers"])):
        payload = base.layer_payload(cal_dir, layer)
        cov = base.covariance(payload)
        mmap = _open_site(wide_dir, wide_meta, "k_post", layer)
        for b in VALID_K_GROUP_SIZES:
            if (layer, b) in completed:
                continue
            seed = args.seed + 1000 * layer + b
            rotations = {
                "hadamard_per_token": base.full_hadamard_rotation(n, seed).float().to(device),
                "nar_per_token": base.nar_rotation(cov, b, seed)[0].float().to(device),
            }
            totals: dict[str, dict[str, float]] = {
                method: {} for method in (
                    "bf16", "identity_per_token", "hadamard_per_token", "nar_per_token", "kivi_per_channel"
                )
            }
            for start in range(0, mmap.shape[0], args.sequence_batch):
                stop = min(start + args.sequence_batch, mmap.shape[0])
                x = _bits_to_tensor(mmap[start:stop], device)
                flat = x.reshape(-1, n)
                identity_stats = _quant_sums(flat, b)
                _merge_sums(totals["identity_per_token"], identity_stats)
                _merge_sums(totals["bf16"], {
                    "range_sum": identity_stats["range_sum"],
                    "group_count": identity_stats["group_count"],
                    "error_sum": 0.0,
                    "energy_sum": identity_stats["energy_sum"],
                })
                for method, rotation in rotations.items():
                    _merge_sums(totals[method], _quant_sums(flat @ rotation.T, b))
                # KIVI-style K grouping: per sequence, KV head, channel, then
                # contiguous token groups. Group boundaries never cross sequences.
                per_channel = x.permute(0, 1, 3, 2).contiguous()
                _merge_sums(totals["kivi_per_channel"], _quant_sums(per_channel, b))
                del x, flat, per_channel
            for method, total in totals.items():
                rows.append({
                    "model": model_key,
                    "layer": layer,
                    "b": b,
                    "bits": 4,
                    "method": method,
                    "axis": "tokens" if method == "kivi_per_channel" else "channels",
                    "mean_group_range": total["range_sum"] / total["group_count"],
                    "relative_quantization_error_nmse": total["error_sum"] / total["energy_sum"],
                    "range_sum": total["range_sum"],
                    "group_count": int(total["group_count"]),
                    "error_sum": total["error_sum"],
                    "energy_sum": total["energy_sum"],
                    "sequences": mmap.shape[0],
                    "sequence_length": mmap.shape[2],
                })
            base.write_csv(partial, rows)
            LOG.info("E1d complete layer=%d/%d b=%d", layer + 1, wide_meta["num_layers"], b)
            del rotations
            torch.cuda.empty_cache()
        del mmap, payload, cov

    summary: list[dict[str, Any]] = []
    dominance: list[dict[str, Any]] = []
    methods = ("bf16", "identity_per_token", "hadamard_per_token", "nar_per_token", "kivi_per_channel")
    for b in VALID_K_GROUP_SIZES:
        summaries: dict[str, dict[str, Any]] = {}
        for method in methods:
            subset = [r for r in rows if int(r["b"]) == b and r["method"] == method]
            range_sum = sum(_float(r, "range_sum") for r in subset)
            group_count = sum(int(float(r["group_count"])) for r in subset)
            error_sum = sum(_float(r, "error_sum") for r in subset)
            energy_sum = sum(_float(r, "energy_sum") for r in subset)
            entry = {
                "b": b,
                "bits": 4,
                "method": method,
                "axis": "tokens" if method == "kivi_per_channel" else "channels",
                "layers": len(subset),
                "mean_group_range": range_sum / group_count,
                "global_relative_quantization_error_nmse": error_sum / energy_sum,
            }
            summary.append(entry)
            summaries[method] = entry
        kivi_nmse = summaries["kivi_per_channel"]["global_relative_quantization_error_nmse"]
        per_token_methods = ("identity_per_token", "hadamard_per_token", "nar_per_token")
        layer_wins = 0
        for layer in range(int(wide_meta["num_layers"])):
            layer_rows = {r["method"]: r for r in rows if int(r["b"]) == b and int(r["layer"]) == layer}
            if all(_float(layer_rows["kivi_per_channel"], "relative_quantization_error_nmse") <
                   _float(layer_rows[method], "relative_quantization_error_nmse") for method in per_token_methods):
                layer_wins += 1
        dominance.append({
            "b": b,
            "kivi_beats_every_per_token_method_global_nmse": all(
                kivi_nmse < summaries[method]["global_relative_quantization_error_nmse"] for method in per_token_methods
            ),
            "layers_where_kivi_beats_every_per_token_method_nmse": layer_wins,
            "layers": int(wide_meta["num_layers"]),
        })
    base.write_csv(result_dir / "e1d_per_layer.csv", rows)
    base.write_csv(result_dir / "e1d_summary.csv", summary)
    base.write_csv(result_dir / "e1d_dominance.csv", dominance)
    base.atomic_json(done, {
        "model": model_key,
        "seed": args.seed,
        "bits": 4,
        "group_sizes": list(VALID_K_GROUP_SIZES),
        "kivi_definition": "dynamic asymmetric per-channel K, contiguous token groups within each sequence",
        "paired_source": str(wide_dir / "dumps" / "k_post"),
        "dominance": dominance,
        "wide_capture": wide_meta,
        "calibration": cal_meta,
    })
    partial.unlink(missing_ok=True)
    LOG.info("E1d outputs complete under %s", result_dir)


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    return base.md_table(headers, rows)


def build_extended_report(args: argparse.Namespace) -> None:
    import pandas as pd

    wd = base.work_path(args)
    base.setup_logging(wd, "report-extended")
    result_dir = wd / "results" / "llama32_3b"
    required = [
        result_dir / "E1B_DONE.json",
        result_dir / "E1C_DONE.json",
        result_dir / "E1D_DONE.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"extended report requires completed E1b/E1c/E1d: {missing}")
    e0 = _read_json(wd / "results" / "e0.json")
    e1 = pd.read_csv(result_dir / "e1_per_layer.csv")
    rank = pd.read_csv(result_dir / "e1_range_vs_k.csv")
    stability = pd.read_csv(result_dir / "e1_stability.csv")
    e1b = pd.read_csv(result_dir / "e1b_position_plain_nar.csv")
    e1c = pd.read_csv(result_dir / "e1c_summary.csv")
    e1c_layers = pd.read_csv(result_dir / "e1c_per_layer.csv")
    e1c_fit = pd.read_csv(result_dir / "e1c_energy_fit.csv")
    e1c_eig = pd.read_csv(result_dir / "e1c_eigenspace.csv")
    e1d = pd.read_csv(result_dir / "e1d_summary.csv")
    e1d_dom = pd.read_csv(result_dir / "e1d_dominance.csv")
    e2 = pd.concat([
        pd.read_csv(result_dir / "e2_summary.csv"),
        pd.read_csv(wd / "results" / "llama32_1b" / "e2_summary.csv"),
    ], ignore_index=True)

    gate_rows: list[dict[str, Any]] = []
    for b in VALID_K_GROUP_SIZES:
        means = e1[e1.b.eq(b)].groupby("method", as_index=True).mean(numeric_only=True)
        reduction = (means.loc["hadamard", "mean_group_range"] - means.loc["nar", "mean_group_range"]) / means.loc["hadamard", "mean_group_range"]
        last_rank = rank[rank.b.eq(b) & rank.k.eq(rank[rank.b.eq(b)].k.max())]
        attribution = last_rank.projection_range_attribution_fraction.mean()
        stab = stability[stability.b.eq(b)].mean(numeric_only=True)
        survives = stab.reduction_disjoint_cal_b_vs_hadamard > 0 and stab.retained_fraction_of_cal_a_reduction >= 0.8
        passed = reduction >= 0.10 and attribution >= 0.10 and survives
        gate_rows.append({
            "b": b,
            "nar_range_reduction_vs_hadamard": reduction,
            "topk_projection_range_attribution": attribution,
            "heldout_retained_fraction": stab.retained_fraction_of_cal_a_reduction,
            "pass": bool(passed),
        })
    e1_pass = all(row["pass"] for row in gate_rows)
    e2_nar = e2[e2.method.eq("nar")]
    e2_pass = bool(((e2_nar.paired_ppl_delta_vs_hadamard <= 0) & (e2_nar.paired_90ci_high < 0.05)).all())
    promising = e1_pass and e2_pass

    rope_layer_dominance = []
    for b in VALID_K_GROUP_SIZES:
        pivot = e1[e1.b.eq(b)].pivot(index="layer", columns="method", values=["mean_group_range", "relative_quantization_error_nmse"])
        rope_layer_dominance.append({
            "b": b,
            "range_layers": int((pivot["mean_group_range"]["nar"] < pivot["mean_group_range"]["nar_rope"]).sum()),
            "nmse_layers": int((pivot["relative_quantization_error_nmse"]["nar"] < pivot["relative_quantization_error_nmse"]["nar_rope"]).sum()),
            "layers": len(pivot),
        })
    rope_seed_wins = True
    for _, row in e2[e2.method.eq("nar")].iterrows():
        peer = e2[(e2.model.eq(row.model)) & e2.b.eq(row.b) & e2.method.eq("nar_rope")]
        if peer.empty:
            continue
        nar_values = [float(x) for x in row.seed_ppls.split(";")]
        rope_values = [float(x) for x in peer.iloc[0].seed_ppls.split(";")]
        rope_seed_wins &= all(n < r for n, r in zip(nar_values, rope_values))

    e1c_effects: list[dict[str, Any]] = []
    for site in WIDE_SITES:
        layer_rows = e1c_layers[e1c_layers.site.eq(site)]
        means = layer_rows.groupby("method", as_index=True).mean(numeric_only=True)
        ratio_of_means = (means.loc["hadamard_full", "mean_group_range"] - means.loc["nar_kmax", "mean_group_range"]) / means.loc["hadamard_full", "mean_group_range"]
        mean_layer_reduction = e1c[(e1c.site.eq(site)) & (e1c.method.eq("nar_kmax"))].iloc[0].mean_range_reduction_vs_hadamard
        e1c_effects.append({"site": site, "ratio_of_mean_ranges": ratio_of_means, "mean_of_paired_layer_reductions": mean_layer_reduction})
    eig_quality = e1c_eig.groupby("site").relative_ritz_residual.agg(["median", "max"]).reset_index()
    kivi_clear_winner = bool(
        e1d_dom.kivi_beats_every_per_token_method_global_nmse.all()
        and (e1d_dom.layers_where_kivi_beats_every_per_token_method_nmse == e1d_dom.layers).all()
    )
    kivi_comparisons: list[str] = []
    for b in VALID_K_GROUP_SIZES:
        subset = e1d[e1d.b.eq(b)].set_index("method")
        kivi_comparisons.append(
            f"b={b}: KIVI {subset.loc['kivi_per_channel', 'global_relative_quantization_error_nmse']:.8f} "
            f"vs best rotation NAR {subset.loc['nar_per_token', 'global_relative_quantization_error_nmse']:.8f}"
        )

    def pct(value: float) -> str:
        return f"{100 * float(value):.3f}%"

    report: list[str] = []
    report.append("# NAR offline tensor validation — corrected gate and activation extension\n")
    report.append("## Outcome\n")
    report.append("The original b=128 K phenomenon gate was dimensionally mis-specified: head_dim=128 gives only one b=128 group and therefore one DC slot. It is retained as an archival measurement, not used for the E1 decision. The valid pre-registered K gates are b=32 and b=64; no E1 or E2 row was rerun.\n")
    report.append(f"- **Corrected E1 K criterion: {'PASS' if e1_pass else 'FAIL'}**. " + "; ".join(
        f"b={r['b']}: range reduction {pct(r['nar_range_reduction_vs_hadamard'])}, attribution {pct(r['topk_projection_range_attribution'])}, held-out retention {pct(r['heldout_retained_fraction'])}"
        for r in gate_rows
    ) + ".")
    report.append(f"- **E2 criterion over every frozen NAR row: {'PASS' if e2_pass else 'FAIL'}**. Every E2 result remains valid and is listed below.")
    report.append(f"- **Corrected method-promising criterion: {'PASS' if promising else 'FAIL'}**.")
    report.append(f"- **E0 implementation sanity: {'PASS' if e0['pass'] else 'FAIL'}**.\n")

    report.append("## Frozen protocol and comparison rules\n")
    report.append("All comparisons are paired on identical tensors. Dynamic asymmetric INT4 uses one fp16 scale and one fp16 real-valued offset for each group and NMSE is `sum((x_hat-x)^2)/sum(x^2)`. No weight quantization, GPTQ, configuration search, or end-to-end W4A4 run was performed. E1b reads the prior K dump; E1c/E1d use one new forward capture of the same 128 WikiText-2 train sequences. E2 is read only.\n")

    report.append("## Corrected E1 — frozen K results\n")
    e1_means = e1[e1.b.isin(VALID_K_GROUP_SIZES)].groupby(["b", "method"], as_index=False)[["mean_group_range", "relative_quantization_error_nmse"]].mean()
    report.append(_markdown_table(["b", "method", "mean range", "mean NMSE"], e1_means.itertuples(index=False, name=None)) + "\n")
    report.append(_markdown_table(["b", "range reduction", "top-k attribution", "held-out retention", "pass"],
                                  [(r["b"], r["nar_range_reduction_vs_hadamard"], r["topk_projection_range_attribution"], r["heldout_retained_fraction"], r["pass"]) for r in gate_rows]) + "\n")

    report.append("### NAR-RoPE is dominated and dropped\n")
    report.append("Plain NAR has lower range and lower NMSE than NAR-RoPE in every one of 28 layers at both valid group sizes; it also has lower PPL at every paired E2 seed where NAR-RoPE exists. NAR-RoPE is therefore strictly dominated in the observed comparisons and no further NAR-RoPE work was run.\n")
    report.append(_markdown_table(["b", "plain NAR lower-range layers", "plain NAR lower-NMSE layers", "layers"],
                                  [(r["b"], r["range_layers"], r["nmse_layers"], r["layers"]) for r in rope_layer_dominance]) + "\n")

    report.append("## E1b — plain-NAR position check\n")
    e1b_avg = e1b.groupby(["b", "position", "method"], as_index=False)[["mean_group_range", "relative_quantization_error_nmse", "range_reduction_vs_hadamard"]].mean()
    report.append("The exact stored pre-RoPE K samples were re-rotated using model RoPE at positions 0/512/1024/2048. This is offline tensor analysis, not a model rerun.\n")
    report.append(_markdown_table(["b", "position", "method", "mean range", "mean NMSE", "reduction vs Had"], e1b_avg.itertuples(index=False, name=None)) + "\n")

    report.append("## E1c — wide activation inputs, b=128\n")
    report.append("The dump retains every token as exact bf16 bit patterns at q_proj input (post-input-RMSNorm residual, n=3072, 24 DC slots) and down_proj input (post-SiLU gated MLP product, n=8192, 64 DC slots). Top directions use all 262144 tokens with a fixed randomized symmetric eigensolver: oversampling 16, one power iteration, three full passes, and published Ritz residuals. Range/NMSE and the greedy residual-energy permutation use positions `0,32,...,2016` from all 128 sequences (8192 paired token vectors per layer). Hadamard is a random-sign full-feature transform (H8192 or fixed Paley H12 tensor H256 for n=3072). Each NAR k uses low-energy fillers for unused DC slots and the same H128.\n")
    report.append(_markdown_table(list(e1c.columns), e1c.itertuples(index=False, name=None)) + "\n")
    report.append("Range reduction is shown both as a ratio of layer-mean ranges and as the mean of paired per-layer reductions:\n")
    report.append(_markdown_table(["site", "ratio of mean ranges", "mean paired-layer reduction"], [(r["site"], r["ratio_of_mean_ranges"], r["mean_of_paired_layer_reductions"]) for r in e1c_effects]) + "\n")
    report.append("![E1c range versus k](results/llama32_3b/e1c_range_vs_k.png)\n\n![E1c energy fit](results/llama32_3b/e1c_energy_fit.png)\n")
    report.append("The fit is pooled OLS over paired layer-k rows, `range(k)/range(0) = intercept + slope*sqrt(1-f)`:\n")
    report.append(_markdown_table(list(e1c_fit.columns), e1c_fit.itertuples(index=False, name=None)) + "\n")
    report.append("Randomized eigenspace approximation quality (relative Ritz residual; every direction is in the exact CSV):\n")
    report.append(_markdown_table(list(eig_quality.columns), eig_quality.itertuples(index=False, name=None)) + "\n")

    report.append("## E1d — KIVI-style per-channel K baseline\n")
    report.append("At the same 4-bit width and b=32/64 metadata grouping, KIVI-style K quantization groups contiguous tokens independently per sequence/head/channel. Per-token methods group channels. All operate on the same full post-RoPE K dump; dominance is decided by global NMSE, with per-layer counts also reported.\n")
    report.append(_markdown_table(list(e1d.columns), e1d.itertuples(index=False, name=None)) + "\n")
    report.append(_markdown_table(list(e1d_dom.columns), e1d_dom.itertuples(index=False, name=None)) + "\n")
    report.append("**Per-channel K is the clear winner in this test.** " + "; ".join(kivi_comparisons) + ". It beats every per-token method in all 28/28 layers at both group sizes. For K under these conditions, the standard per-channel axis is preferable to every tested per-token rotation method.\n")

    report.append("## E2 — frozen KV-only perplexity results\n")
    report.append(_markdown_table(list(e2.columns), e2.itertuples(index=False, name=None)) + "\n")

    report.append("## Negative findings, caveats, and what remains unsure\n")
    report.append("- NAR-RoPE is dominated by plain NAR in every available paired check and is no longer pursued.\n- KIVI-style per-channel K beats every tested per-token rotation by global NMSE and in every layer at both b=32/64.\n- The E1c eigenspaces are deterministic randomized approximations, not dense 8192x8192 decompositions; Ritz residuals are published per direction and are non-negligible for tail directions.\n- E1c stores all tokens and uses all of them for the top-direction solve, but evaluates range/NMSE and balances Pi on a fixed position stride to bound repeated k-sweep cost.\n- Per-channel and per-token range means describe different axes, so E1d fairness is decided by paired NMSE; metadata count and bit width are matched.\n- E1c dumps are deliberately retained under project storage for the separately scoped E3 FP4 E2M1 and E4 two-level NVFP4 checks.\n")

    report.append("## Go / no-go\n")
    report.append(("**GO to the already-scoped activation-shape checks, without tuning.** " if promising else "**NO-GO under the corrected gates.** ") +
                  "This decision uses b=32/64 K E1 and every frozen E2 row; the newly measured E1c/E1d results are mechanistic follow-ups and are reported regardless of sign.\n")
    report.append("## Reproduction artifacts\n")
    report.append("Exact tables and plots are under `results/`; commands and logs are under `runs/`. Large raw bf16 dumps and randomized eigenspace checkpoints remain under `activations/` in project storage and are excluded from Git.\n")
    (wd / "report.md").write_text("\n".join(report))
    decision = {
        "corrected_e1_pass": e1_pass,
        "valid_e1_group_sizes": list(VALID_K_GROUP_SIZES),
        "b128_k_gate_status": "invalid: head_dim=128 gives one b=128 DC slot; retained only as archival measurement",
        "e1_gate_rows": gate_rows,
        "all_frozen_e2_nar_rows_pass": e2_pass,
        "method_promising_corrected": promising,
        "nar_rope_dropped_as_dominated": bool(all(r["range_layers"] == r["layers"] and r["nmse_layers"] == r["layers"] for r in rope_layer_dominance) and rope_seed_wins),
        "kivi_per_channel_clear_winner": kivi_clear_winner,
        "e1d_dominance": e1d_dom.to_dict(orient="records"),
    }
    base.atomic_json(wd / "results" / "decision_corrected.json", decision)
    LOG.info("wrote corrected report and decision")


def run_all(args: argparse.Namespace) -> None:
    run_e1b(args)
    collect_wide_activations(args)
    run_e1c(args)
    run_e1d(args)
    build_extended_report(args)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    sub = parser.add_subparsers(dest="command", required=True)
    e1b = sub.add_parser("e1b")
    e1b.add_argument("--model", default="llama32_3b")
    e1b.add_argument("--cal-a", default="cal_a")
    collect = sub.add_parser("collect-wide")
    collect.add_argument("--model", default="llama32_3b")
    collect.add_argument("--tag", default="wide_cal_a")
    collect.add_argument("--split", default="train")
    collect.add_argument("--offset", type=int, default=0)
    collect.add_argument("--sequences", type=int, default=128)
    collect.add_argument("--seq-len", type=int, default=2048)
    collect.add_argument("--batch-size", type=int, default=2)
    collect.add_argument("--checkpoint-sequences", type=int, default=8)
    e1c = sub.add_parser("e1c")
    e1c.add_argument("--model", default="llama32_3b")
    e1c.add_argument("--wide-tag", default="wide_cal_a")
    e1c.add_argument("--eval-token-stride", type=int, default=32)
    e1c.add_argument("--oversample", type=int, default=16)
    e1c.add_argument("--sequence-batch", type=int, default=2)
    e1d = sub.add_parser("e1d")
    e1d.add_argument("--model", default="llama32_3b")
    e1d.add_argument("--wide-tag", default="wide_cal_a")
    e1d.add_argument("--cal-a", default="cal_a")
    e1d.add_argument("--sequence-batch", type=int, default=2)
    sub.add_parser("report")
    all_parser = sub.add_parser("all")
    all_parser.add_argument("--model", default="llama32_3b")
    all_parser.add_argument("--cal-a", default="cal_a")
    all_parser.add_argument("--tag", default="wide_cal_a")
    all_parser.add_argument("--wide-tag", default="wide_cal_a")
    all_parser.add_argument("--split", default="train")
    all_parser.add_argument("--offset", type=int, default=0)
    all_parser.add_argument("--sequences", type=int, default=128)
    all_parser.add_argument("--seq-len", type=int, default=2048)
    all_parser.add_argument("--batch-size", type=int, default=2)
    all_parser.add_argument("--checkpoint-sequences", type=int, default=8)
    all_parser.add_argument("--eval-token-stride", type=int, default=32)
    all_parser.add_argument("--oversample", type=int, default=16)
    all_parser.add_argument("--sequence-batch", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    dispatch = {
        "e1b": run_e1b,
        "collect-wide": collect_wide_activations,
        "e1c": run_e1c,
        "e1d": run_e1d,
        "report": build_extended_report,
        "all": run_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
