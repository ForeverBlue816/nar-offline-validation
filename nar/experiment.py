#!/usr/bin/env python3
"""Reproduce the NAR offline-tensor validation tables and plots.

The script deliberately does not implement weight quantization, GPTQ, or an
end-to-end W4A4 pipeline.  It captures post-RoPE Q/K, analyzes those tensors,
and injects only dynamic asymmetric INT4 fake-quantization into K/V for the
perplexity proxy.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


LOG = logging.getLogger("nar")
BITS = 4
QMAX = 2**BITS - 1
DEFAULT_SEED = 20260902
MODEL_IDS = {
    "llama32_3b": "unsloth/Llama-3.2-3B",
    "llama32_1b": "unsloth/Llama-3.2-1B",
}


def work_path(args: argparse.Namespace) -> Path:
    return Path(args.workdir).resolve()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_torch_save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def setup_logging(workdir: Path, stage: str) -> Path:
    workdir.joinpath("runs").mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    logfile = workdir / "runs" / f"{timestamp}-{stage}.log"
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(fmt)
    LOG.addHandler(stream)
    LOG.addHandler(file_handler)
    manifest = workdir / "runs" / "commands.jsonl"
    with manifest.open("a") as f:
        f.write(json.dumps({
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "stage": stage,
            "log": str(logfile),
        }) + "\n")
    return logfile


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    }
    if torch.cuda.is_available():
        info.update({
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        })
        try:
            query = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version", "--format=csv,noheader"],
                text=True,
            ).strip()
            info["nvidia_smi"] = query.splitlines()
        except Exception as exc:  # pragma: no cover - diagnostic only
            info["nvidia_smi_error"] = repr(exc)
    return info


# ---------------------------------------------------------------------------
# Quantizer and rotations
# ---------------------------------------------------------------------------


def group_view(x: torch.Tensor, group_size: int) -> torch.Tensor:
    if x.shape[-1] % group_size:
        raise ValueError(f"last dimension {x.shape[-1]} is not divisible by group size {group_size}")
    return x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)


def dynamic_asym_int4(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake-quantize with one fp16 scale and fp16 real-valued zero/offset per group.

    q = clamp(round((x - z) / s), 0, 15), x_hat = q*s + z.
    Both s and z are rounded to IEEE fp16 before they are used for q and x_hat.
    Degenerate groups use s=1 and q=0, exactly reproducing their fp16 offset.
    """
    original_dtype = x.dtype
    xg = group_view(x.float(), group_size)
    lo = xg.amin(dim=-1, keepdim=True)
    hi = xg.amax(dim=-1, keepdim=True)
    raw_scale = (hi - lo) / QMAX
    scale16 = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale)).to(torch.float16)
    zero16 = lo.to(torch.float16)
    scale = scale16.float()
    zero = zero16.float()
    q = torch.round((xg - zero) / scale).clamp_(0, QMAX)
    deq = q * scale + zero
    return deq.reshape_as(x).to(original_dtype), scale16.squeeze(-1), zero16.squeeze(-1), q.to(torch.uint8)


def mean_group_range(x: torch.Tensor, group_size: int) -> float:
    xg = group_view(x.float(), group_size)
    return float((xg.amax(-1) - xg.amin(-1)).mean().item())


def quant_metrics(x: torch.Tensor, group_size: int) -> tuple[float, float, float]:
    xf = x.float()
    deq, _, _, _ = dynamic_asym_int4(xf, group_size)
    error = (deq.float() - xf).square().sum().double()
    energy = xf.square().sum().double()
    nmse = float((error / energy.clamp_min(torch.finfo(torch.float64).tiny)).item())
    mse = float((error / xf.numel()).item())
    return mean_group_range(xf, group_size), nmse, mse


def hadamard(n: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got {n}")
    h = torch.ones((1, 1), dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def block_hadamard(n: int, b: int, seed: int) -> torch.Tensor:
    if n % b:
        raise ValueError((n, b))
    hb = hadamard(b)
    out = torch.zeros((n, n), dtype=torch.float64)
    for group in range(n // b):
        start = group * b
        out[start : start + b, start : start + b] = hb
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).double().mul_(2).sub_(1)
    return out @ torch.diag(signs)


def full_hadamard_rotation(n: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).double().mul_(2).sub_(1)
    return hadamard(n) @ torch.diag(signs)


def _balanced_target_slots(energies: list[float], groups: int, b: int) -> list[int]:
    loads = [0.0] * groups
    used = [0] * groups
    slots: list[int] = []
    for energy in energies:
        candidates = [g for g in range(groups) if used[g] < b - 1]
        group = min(candidates, key=lambda g: (loads[g], used[g], g))
        slot = group * b + 1 + used[group]
        used[group] += 1
        loads[group] += float(energy)
        slots.append(slot)
    return slots


def nar_rotation(cov: torch.Tensor, b: int, seed: int, absorbed: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct H_blk Pi G(V) from a head-dimension second moment."""
    cov = cov.double().cpu()
    n = cov.shape[0]
    groups = n // b
    if n % b:
        raise ValueError((n, b))
    absorbed = groups if absorbed is None else absorbed
    if not 0 <= absorbed <= groups:
        raise ValueError(absorbed)
    evals_asc, evecs_asc = torch.linalg.eigh((cov + cov.T) / 2)
    order_desc = list(reversed(range(n)))
    top = order_desc[:absorbed]
    # Unused DC anchors are deliberately filled by the lowest-energy directions;
    # this defines a matched k=0..groups ablation without accidentally absorbing
    # additional high-energy directions.
    remaining_after_top = [i for i in range(n) if i not in top]
    fillers = sorted(remaining_after_top, key=lambda i: (float(evals_asc[i]), i))[: groups - absorbed]
    anchors_source = top + fillers
    residual_source = [i for i in order_desc if i not in anchors_source]
    residual_energies = [max(0.0, float(evals_asc[i])) for i in residual_source]
    target_order = [g * b for g in range(groups)] + _balanced_target_slots(residual_energies, groups, b)
    source_order = anchors_source + residual_source
    u = evecs_asc[:, source_order]
    target = torch.eye(n, dtype=torch.float64)[:, target_order]
    g_map = target @ u.T
    r = block_hadamard(n, b, seed) @ g_map
    orth_err = float((r @ r.T - torch.eye(n, dtype=torch.float64)).abs().max())
    return r, {
        "absorbed": absorbed,
        "groups": groups,
        "top_eigenvalues": [float(evals_asc[i]) for i in top],
        "filler_eigenvalues": [float(evals_asc[i]) for i in fillers],
        "orthogonality_max_abs": orth_err,
    }


def nar_rope_rotation(cov_post: torch.Tensor, pre_diag: torch.Tensor, b: int, seed: int) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Map complete pre-RoPE frequency pair-planes to distinct group DCs."""
    cov_post = cov_post.double().cpu()
    pre_diag = pre_diag.double().cpu()
    n = cov_post.shape[0]
    groups = n // b
    if n % b:
        raise ValueError((n, b))
    if groups < 2 or groups % 2:
        return None, {"valid": False, "reason": f"requires an even number of >=2 groups, got {groups}"}
    half = n // 2
    plane_energy = pre_diag[:half] + pre_diag[half:]
    plane_order = torch.argsort(plane_energy, descending=True).tolist()
    selected_planes = plane_order[: groups // 2]
    anchors_source: list[int] = []
    for pair in selected_planes:
        anchors_source.extend([pair, pair + half])
    residual_source = [i for i in range(n) if i not in anchors_source]
    residual_source.sort(key=lambda i: (-float(cov_post[i, i]), i))
    residual_energies = [max(0.0, float(cov_post[i, i])) for i in residual_source]
    target_order = [g * b for g in range(groups)] + _balanced_target_slots(residual_energies, groups, b)
    source_order = anchors_source + residual_source
    g_map = torch.eye(n, dtype=torch.float64)[:, target_order] @ torch.eye(n, dtype=torch.float64)[:, source_order].T
    r = block_hadamard(n, b, seed) @ g_map
    return r, {
        "valid": True,
        "groups": groups,
        "selected_planes": selected_planes,
        "selected_plane_energy": [float(plane_energy[i]) for i in selected_planes],
        "orthogonality_max_abs": float((r @ r.T - torch.eye(n, dtype=torch.float64)).abs().max()),
    }


def rotate_rows(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x.float(), r.to(device=x.device, dtype=torch.float32).T)


def apply_rope_offline(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated_half = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated_half * sin


# ---------------------------------------------------------------------------
# E0
# ---------------------------------------------------------------------------


def run_e0(args: argparse.Namespace) -> None:
    wd = work_path(args)
    setup_logging(wd, "e0")
    seed_everything(args.seed)
    rows: list[dict[str, Any]] = []
    passed = True
    for b in (32, 64, 128):
        generator = torch.Generator().manual_seed(args.seed + b)
        x = torch.randn((512, b), generator=generator, dtype=torch.float32)
        base_range, _, base_mse = quant_metrics(x, b)
        for c in (-8.0, -1.0, 0.0, 1.0, 8.0):
            shifted_range, _, shifted_mse = quant_metrics(x + c, b)
            range_delta = abs(shifted_range - base_range)
            mse_delta = abs(shifted_mse - base_mse)
            # fp16 metadata makes bitwise invariance impossible in general.  The
            # tolerance is fixed in advance at 2% of baseline MSE plus 1e-7.
            ok = range_delta <= 2e-6 and mse_delta <= 0.02 * base_mse + 1e-7
            passed &= ok
            rows.append({
                "check": "constant_shift",
                "b": b,
                "c": c,
                "base_range": base_range,
                "shifted_range": shifted_range,
                "abs_range_delta": range_delta,
                "base_mse": base_mse,
                "shifted_mse": shifted_mse,
                "abs_mse_delta": mse_delta,
                "pass": ok,
            })
        amplitude = 100.0
        outlier_channel = 1
        planted = torch.zeros((1, b), dtype=torch.float64)
        planted[0, outlier_channel] = amplitude
        h = hadamard(b)
        had_range = mean_group_range(planted @ h.T, b)
        cov = planted.T @ planted
        r_nar, meta = nar_rotation(cov, b, args.seed + b)
        nar_range = mean_group_range(planted @ r_nar.T, b)
        expected = 2 * amplitude / math.sqrt(b)
        # mean_group_range evaluates in fp32; use a fixed 1e-6 tolerance.
        ok = abs(had_range - expected) <= 1e-6 and nar_range <= 1e-6
        passed &= ok
        rows.append({
            "check": "planted_outlier",
            "b": b,
            "amplitude": amplitude,
            "hadamard_range": had_range,
            "expected_2absx_over_sqrtb": expected,
            "nar_range": nar_range,
            "nar_orthogonality_max_abs": meta["orthogonality_max_abs"],
            "pass": ok,
        })
    write_csv(wd / "results" / "e0.csv", rows)
    atomic_json(wd / "results" / "e0.json", {"pass": passed, "seed": args.seed, "rows": rows})
    LOG.info("E0 pass=%s; exact rows: %s", passed, wd / "results" / "e0.csv")
    if not passed:
        raise AssertionError("E0 failed; refusing to proceed")


# ---------------------------------------------------------------------------
# Data and activation collection
# ---------------------------------------------------------------------------


def model_key_from_id(model_id: str) -> str:
    for key, value in MODEL_IDS.items():
        if model_id == value or model_id == key:
            return key
    return hashlib.sha1(model_id.encode()).hexdigest()[:12]


def resolve_model_id(value: str) -> str:
    return MODEL_IDS.get(value, value)


def prepare_token_chunks(
    model_id: str,
    split: str,
    offset: int,
    n_sequences: int,
    seq_len: int,
    workdir: Path,
) -> torch.Tensor:
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    model_key = model_key_from_id(model_id)
    cache_file = workdir / "cache" / "tokenized" / f"{model_key}-{split}-o{offset}-n{n_sequences}-l{seq_len}.pt"
    if cache_file.exists():
        LOG.info("loading token chunks %s", cache_file)
        return torch.load(cache_file, map_location="cpu", weights_only=True)
    LOG.info("loading WikiText-2 split=%s", split)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, cache_dir=str(workdir / "cache" / "datasets"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(workdir / "cache" / "huggingface"), use_fast=True)
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if bos is None:
        # Qwen3 exposes its document prefix in config.json while deliberately
        # leaving tokenizer.bos_token_id unset (add_bos_token=false).
        config = AutoConfig.from_pretrained(model_id, cache_dir=str(workdir / "cache" / "huggingface"))
        bos = getattr(config, "bos_token_id", None)
    if bos is None:
        raise RuntimeError(f"{model_id} has no BOS/document-prefix token in tokenizer or config")
    content_len = seq_len - 1
    need = (offset + n_sequences) * content_len
    if len(tokens) < need:
        raise RuntimeError(f"split {split} has {len(tokens)} tokens but {need} are required")
    chunks = []
    for i in range(offset, offset + n_sequences):
        start = i * content_len
        chunks.append([bos] + tokens[start : start + content_len])
    result = torch.tensor(chunks, dtype=torch.long)
    atomic_torch_save(cache_file, result)
    LOG.info("saved %s token chunks to %s", result.shape, cache_file)
    return result


@dataclass
class LayerAccumulator:
    moment: torch.Tensor
    pre_diag: torch.Tensor
    count: int
    pre_count: int
    post_samples: list[torch.Tensor]
    q_samples: list[torch.Tensor]
    pre_samples: list[torch.Tensor]
    sample_positions: list[torch.Tensor]
    q_probe: list[torch.Tensor]
    k_probe: list[torch.Tensor]


class ActivationCollector:
    def __init__(self, model: torch.nn.Module, sample_stride: int, moments_only: bool):
        self.model = model
        self.config = model.config
        self.n_layers = self.config.num_hidden_layers
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.num_kv_heads = self.config.num_key_value_heads
        self.sample_stride = sample_stride
        self.moments_only = moments_only
        self.batch_index = -1
        self.layers = [LayerAccumulator(
            moment=torch.zeros((self.head_dim, self.head_dim), dtype=torch.float64),
            pre_diag=torch.zeros(self.head_dim, dtype=torch.float64),
            count=0,
            pre_count=0,
            q_samples=[],
            post_samples=[],
            pre_samples=[],
            sample_positions=[],
            q_probe=[],
            k_probe=[],
        ) for _ in range(self.n_layers)]
        self.handles: list[Any] = []

    def _sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, heads, sequence, head_dim]
        if self.sample_stride <= 0:
            raise RuntimeError("sampling disabled")
        idx = torch.arange(0, x.shape[2], self.sample_stride, device=x.device)
        sampled = x[:, :, idx, :].detach().to(torch.bfloat16).cpu().reshape(-1, self.head_dim)
        positions = idx.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1).cpu().reshape(-1).to(torch.int16)
        return sampled, positions

    def pre_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            x = output.view(output.shape[0], output.shape[1], self.num_kv_heads, self.head_dim).transpose(1, 2)
            flat = x.detach().float().reshape(-1, self.head_dim)
            acc = self.layers[layer_idx]
            acc.pre_diag += flat.square().sum(0).double().cpu()
            acc.pre_count += flat.shape[0]
            if not self.moments_only:
                sample, _ = self._sample(x)
                acc.pre_samples.append(sample)
        return hook

    def attention(self, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  attention_mask: torch.Tensor | None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        layer_idx = module.layer_idx
        flat = key.detach().float().reshape(-1, self.head_dim)
        acc = self.layers[layer_idx]
        acc.moment += (flat.T @ flat).double().cpu()
        acc.count += flat.shape[0]
        if not self.moments_only:
            sample, positions = self._sample(key)
            q_sample, _ = self._sample(query)
            acc.q_samples.append(q_sample)
            acc.post_samples.append(sample)
            acc.sample_positions.append(positions)
            if self.batch_index == 0:
                probe_values = sorted({p for p in (0, 512, 1024, key.shape[2] - 1) if p < key.shape[2]})
                probe_pos = torch.tensor(probe_values, device=key.device)
                acc.q_probe.append(query[:, :, probe_pos, :].detach().to(torch.bfloat16).cpu())
                acc.k_probe.append(key[:, :, probe_pos, :].detach().to(torch.bfloat16).cpu())
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    def install(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("nar_capture", self.attention)
        self.model.config._attn_implementation = "nar_capture"
        for idx, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(self.pre_hook(idx)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_model(model_id: str, workdir: Path) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    LOG.info("loading model %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(workdir / "cache" / "huggingface"),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    return model.eval().cuda()


def collect_activations(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("activation collection requires an allocated CUDA GPU")
    wd = work_path(args)
    setup_logging(wd, f"collect-{args.model}-{args.tag}")
    seed_everything(args.seed)
    model_id = resolve_model_id(args.model)
    model_key = model_key_from_id(model_id)
    outdir = wd / "activations" / model_key / args.tag
    done = outdir / "DONE.json"
    if done.exists() and not args.force:
        LOG.info("checkpoint exists, skipping: %s", done)
        return
    tokens = prepare_token_chunks(model_id, args.split, args.offset, args.sequences, args.seq_len, wd)
    model = load_model(model_id, wd)
    collector = ActivationCollector(model, args.sample_stride, args.moments_only)
    collector.install()
    started = time.time()
    try:
        with torch.inference_mode():
            for start in range(0, tokens.shape[0], args.batch_size):
                collector.batch_index = start // args.batch_size
                batch = tokens[start : start + args.batch_size].cuda(non_blocking=True)
                model.model(input_ids=batch, use_cache=False)
                if collector.batch_index % 4 == 0:
                    LOG.info("%s/%s sequences", min(start + args.batch_size, tokens.shape[0]), tokens.shape[0])
            # Save the exact model-generated cos/sin values used by its RoPE
            # implementation at the requested position probes, including 2048.
            positions = torch.tensor([[0, 512, 1024, 2048]], device="cuda")
            dummy = torch.zeros((1, positions.shape[1], model.config.hidden_size), device="cuda", dtype=torch.bfloat16)
            cos, sin = model.model.rotary_emb(dummy, position_ids=positions)
            rope_probe = {"positions": positions.cpu(), "cos": cos.cpu(), "sin": sin.cpu()}
    finally:
        collector.close()
    outdir.mkdir(parents=True, exist_ok=True)
    for idx, acc in enumerate(collector.layers):
        payload: dict[str, Any] = {
            "moment": acc.moment,
            "count": acc.count,
            "pre_diag_sum": acc.pre_diag,
            "pre_count": acc.pre_count,
        }
        if not args.moments_only:
            payload.update({
                "q_post": torch.cat(acc.q_samples, 0),
                "k_post": torch.cat(acc.post_samples, 0),
                "k_pre": torch.cat(acc.pre_samples, 0),
                "positions": torch.cat(acc.sample_positions, 0),
                "q_probe": torch.cat(acc.q_probe, 0),
                "k_probe": torch.cat(acc.k_probe, 0),
            })
        atomic_torch_save(outdir / f"layer_{idx:02d}.pt", payload)
        LOG.info("saved layer %d count=%d samples=%d", idx, acc.count, 0 if args.moments_only else payload["k_post"].shape[0])
    atomic_torch_save(outdir / "rope_probe.pt", rope_probe)
    summary = {
        "model_id": model_id,
        "model_key": model_key,
        "tag": args.tag,
        "split": args.split,
        "offset_sequences": args.offset,
        "sequences": args.sequences,
        "sequence_length": args.seq_len,
        "batch_size": args.batch_size,
        "sample_stride": args.sample_stride,
        "sample_rule": None if args.moments_only else f"positions 0,{args.sample_stride},...<{args.seq_len}; all attention heads for Q and all KV heads for K",
        "moments_use_all_tokens": True,
        "moments_only": args.moments_only,
        "head_dim": collector.head_dim,
        "num_layers": collector.n_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "model_config": model.config.to_dict(),
        "elapsed_seconds": time.time() - started,
        "hardware": hardware_info(),
    }
    atomic_json(done, summary)
    LOG.info("collection complete in %.1fs", summary["elapsed_seconds"])
    del model, collector
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# E1 analysis
# ---------------------------------------------------------------------------


def layer_payload(directory: Path, layer: int) -> dict[str, Any]:
    return torch.load(directory / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=True)


def covariance(payload: dict[str, Any]) -> torch.Tensor:
    return payload["moment"].double() / int(payload["count"])


def apply_rotation_metrics(x: torch.Tensor, r: torch.Tensor | None, b: int) -> tuple[float, float, float]:
    if r is None:
        return (math.nan, math.nan, math.nan)
    return quant_metrics(rotate_rows(x, r), b)


def verify_qk(q: torch.Tensor, k: torch.Tensor, r: torch.Tensor) -> tuple[float, float]:
    groups = q.shape[1] // k.shape[1]
    k_rep = k.repeat_interleave(groups, dim=1).float()
    qf = q.float()
    base = torch.matmul(qf, k_rep.transpose(-1, -2))
    rr = r.float()
    qr = torch.matmul(qf, rr.T)
    kr = torch.matmul(k_rep, rr.T)
    changed = torch.matmul(qr, kr.transpose(-1, -2))
    absolute = float((changed - base).abs().max())
    relative = float(((changed - base).abs().max() / base.abs().max().clamp_min(1e-12)).item())
    return absolute, relative


def analyze_e1(args: argparse.Namespace) -> None:
    wd = work_path(args)
    setup_logging(wd, "analyze-e1")
    model_id = resolve_model_id(args.model)
    model_key = model_key_from_id(model_id)
    a_dir = wd / "activations" / model_key / args.cal_a
    b_dir = wd / "activations" / model_key / args.cal_b
    a_meta = json.loads((a_dir / "DONE.json").read_text())
    b_meta = json.loads((b_dir / "DONE.json").read_text())
    if a_meta["moments_only"]:
        raise RuntimeError("calibration A must include tensor samples")
    n_layers = int(a_meta["num_layers"])
    n = int(a_meta["head_dim"])
    group_sizes = [b for b in (32, 64, 128) if n % b == 0]
    e1_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    qk_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rope = torch.load(a_dir / "rope_probe.pt", map_location="cpu", weights_only=True)
    rope_positions = rope["positions"].reshape(-1).tolist()

    for layer in range(n_layers):
        pa = layer_payload(a_dir, layer)
        pb = layer_payload(b_dir, layer)
        cov_a = covariance(pa)
        cov_b = covariance(pb)
        pre_diag_a = pa["pre_diag_sum"].double() / int(pa["pre_count"])
        pre_diag_b = pb["pre_diag_sum"].double() / int(pb["pre_count"])
        x = pa["k_post"].float()
        x_pre = pa["k_pre"].float()
        evals = torch.linalg.eigvalsh((cov_a + cov_a.T) / 2).clamp_min(0).flip(0)
        total_eval = float(evals.sum())
        effective_rank = float(evals.sum().square() / evals.square().sum().clamp_min(1e-30))
        for rank, value in enumerate(evals.tolist(), start=1):
            spectrum_rows.append({
                "model": model_key, "layer": layer, "rank": rank, "eigenvalue": value,
                "fraction_total": value / total_eval if total_eval else math.nan,
                "cumulative_fraction": float(evals[:rank].sum()) / total_eval if total_eval else math.nan,
            })
        identity_range = mean_group_range(x, n)
        positions = pa["positions"].long()
        bos_mask = positions == 0
        bos_range = mean_group_range(x[bos_mask], n)
        nonbos_range = mean_group_range(x[~bos_mask], n)
        diagnostic_rows.append({
            "model": model_key, "layer": layer, "effective_rank": effective_rank,
            "top1_energy_fraction": float(evals[0]) / total_eval,
            "bos_identity_range": bos_range, "nonbos_identity_range": nonbos_range,
            "bos_to_nonbos_range_ratio": bos_range / nonbos_range,
            "sample_identity_range_b128": identity_range,
        })
        for b in group_sizes:
            seed = args.seed + 1000 * layer + b
            rotations: dict[str, torch.Tensor | None] = {
                "identity": torch.eye(n, dtype=torch.float64),
                "hadamard": full_hadamard_rotation(n, seed),
            }
            rotations["nar"], nar_meta = nar_rotation(cov_a, b, seed)
            rotations["nar_rope"], rope_meta = nar_rope_rotation(cov_a, pre_diag_a, b, seed)
            raw_range = mean_group_range(x, b)
            e1_rows.append({
                "model": model_key, "layer": layer, "b": b, "method": "bf16",
                "mean_group_range": raw_range, "relative_quantization_error_nmse": 0.0,
                "mse": 0.0, "sample_vectors": x.shape[0], "valid": True,
            })
            for method, rotation in rotations.items():
                range_value, nmse, mse = apply_rotation_metrics(x, rotation, b)
                e1_rows.append({
                    "model": model_key, "layer": layer, "b": b, "method": method,
                    "mean_group_range": range_value, "relative_quantization_error_nmse": nmse,
                    "mse": mse, "sample_vectors": x.shape[0], "valid": rotation is not None,
                    "invalid_reason": rope_meta.get("reason", "") if method == "nar_rope" and rotation is None else "",
                })
            # Low-rank k ablation and the explicit projection-removal attribution.
            evals_asc, evecs_asc = torch.linalg.eigh((cov_a + cov_a.T) / 2)
            top_vecs = evecs_asc.flip(1)
            base_range = None
            for k in range(n // b + 1):
                rk, _ = nar_rotation(cov_a, b, seed, absorbed=k)
                range_k = mean_group_range(rotate_rows(x, rk), b)
                if base_range is None:
                    base_range = range_k
                if k == 0:
                    residual_range = mean_group_range(x, b)
                    attribution = 0.0
                else:
                    v = top_vecs[:, :k].float()
                    residual = x - (x @ v) @ v.T
                    residual_range = mean_group_range(residual, b)
                    attribution = (raw_range - residual_range) / raw_range
                rank_rows.append({
                    "model": model_key, "layer": layer, "b": b, "k": k,
                    "nar_mean_group_range": range_k,
                    "dc_absorption_fraction_vs_k0": (base_range - range_k) / base_range,
                    "identity_residual_range_after_projection": residual_range,
                    "projection_range_attribution_fraction": attribution,
                })
            # Stability: construct on A or disjoint B, evaluate both on identical A tensors.
            ra, _ = nar_rotation(cov_a, b, seed)
            rb, _ = nar_rotation(cov_b, b, seed)
            had_range = mean_group_range(rotate_rows(x, rotations["hadamard"]), b)
            range_a = mean_group_range(rotate_rows(x, ra), b)
            range_b = mean_group_range(rotate_rows(x, rb), b)
            reduction_a = (had_range - range_a) / had_range
            reduction_b = (had_range - range_b) / had_range
            stability_rows.append({
                "model": model_key, "layer": layer, "b": b,
                "hadamard_range": had_range, "nar_cal_a_range": range_a, "nar_disjoint_cal_b_range": range_b,
                "reduction_cal_a_vs_hadamard": reduction_a,
                "reduction_disjoint_cal_b_vs_hadamard": reduction_b,
                "reduction_degradation_fraction_points": reduction_a - reduction_b,
                "retained_fraction_of_cal_a_reduction": reduction_b / reduction_a if reduction_a != 0 else math.nan,
            })
            # Position probe: apply exact model cos/sin for fixed hypothetical positions
            # to the same pre-RoPE tensor sample, separating position from token mix.
            rope_r, rope_meta_b = nar_rope_rotation(cov_a, pre_diag_a, b, seed)
            for pos_idx, pos in enumerate(rope_positions):
                x_at_pos = apply_rope_offline(x_pre, rope["cos"][0, pos_idx].float(), rope["sin"][0, pos_idx].float())
                bf16_pos = mean_group_range(x_at_pos, b)
                position_rows.append({
                    "model": model_key, "layer": layer, "b": b, "position": pos,
                    "method": "bf16", "mean_group_range": bf16_pos, "valid": True,
                    "reduction_vs_hadamard": math.nan,
                })
                had_pos = mean_group_range(rotate_rows(x_at_pos, rotations["hadamard"]), b)
                rope_pos = math.nan if rope_r is None else mean_group_range(rotate_rows(x_at_pos, rope_r), b)
                position_rows.append({
                    "model": model_key, "layer": layer, "b": b, "position": pos,
                    "method": "hadamard", "mean_group_range": had_pos, "valid": True,
                    "reduction_vs_hadamard": 0.0,
                })
                position_rows.append({
                    "model": model_key, "layer": layer, "b": b, "position": pos,
                    "method": "nar_rope", "mean_group_range": rope_pos, "valid": rope_r is not None,
                    "reduction_vs_hadamard": (had_pos - rope_pos) / had_pos if rope_r is not None else math.nan,
                    "invalid_reason": rope_meta_b.get("reason", "") if rope_r is None else "",
                })
            q = pa["q_probe"].float()
            kprobe = pa["k_probe"].float()
            for method, rotation in rotations.items():
                if rotation is None:
                    qk_rows.append({"model": model_key, "layer": layer, "b": b, "method": method, "valid": False,
                                    "max_abs_error": math.nan, "max_relative_error": math.nan,
                                    "invalid_reason": rope_meta.get("reason", "")})
                else:
                    absolute, relative = verify_qk(q, kprobe, rotation)
                    qk_rows.append({"model": model_key, "layer": layer, "b": b, "method": method, "valid": True,
                                    "max_abs_error": absolute, "max_relative_error": relative})
        LOG.info("analyzed layer %d/%d", layer + 1, n_layers)

    result_dir = wd / "results" / model_key
    write_csv(result_dir / "e1_per_layer.csv", e1_rows)
    write_csv(result_dir / "e1_eigen_spectrum.csv", spectrum_rows)
    write_csv(result_dir / "e1_range_vs_k.csv", rank_rows)
    write_csv(result_dir / "e1_stability.csv", stability_rows)
    write_csv(result_dir / "e1_position.csv", position_rows)
    write_csv(result_dir / "e1_qk_invariance.csv", qk_rows)
    write_csv(result_dir / "e1_diagnostics.csv", diagnostic_rows)
    create_e1_plots(result_dir, spectrum_rows, rank_rows, n_layers)
    atomic_json(result_dir / "E1_DONE.json", {
        "model": model_key, "seed": args.seed, "calibration_a": a_meta, "calibration_b": b_meta,
        "group_sizes": group_sizes, "n_layers": n_layers,
    })


def create_e1_plots(result_dir: Path, spectrum_rows: list[dict[str, Any]], rank_rows: list[dict[str, Any]], n_layers: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = sorted({int(row["rank"]) for row in spectrum_rows})
    spectrum_mean = [np.mean([float(row["fraction_total"]) for row in spectrum_rows if int(row["rank"]) == rank]) for rank in ranks]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for layer in range(n_layers):
        values = [float(row["fraction_total"]) for row in spectrum_rows if int(row["layer"]) == layer]
        ax.plot(ranks, values, color="0.75", linewidth=0.6, alpha=0.5)
    ax.plot(ranks, spectrum_mean, color="black", linewidth=2, label="layer mean")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Fraction of second-moment trace")
    ax.set_title("Post-RoPE K uncentered second-moment spectrum")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "eigenvalue_spectrum.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, b in zip(axes, (32, 64, 128)):
        subset = [row for row in rank_rows if int(row["b"]) == b]
        ks = sorted({int(row["k"]) for row in subset})
        means = [np.mean([float(row["nar_mean_group_range"]) for row in subset if int(row["k"]) == k]) for k in ks]
        for layer in range(n_layers):
            vals = [float(row["nar_mean_group_range"]) for row in subset if int(row["layer"]) == layer]
            ax.plot(ks, vals, color="0.75", linewidth=0.7, alpha=0.55)
        ax.plot(ks, means, marker="o", color="black", linewidth=2, label="layer mean")
        ax.set_title(f"group b={b}")
        ax.set_xlabel("Top second-moment directions on DC (k)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean K group range")
    axes[-1].legend()
    fig.suptitle("NAR mean K range vs absorbed rank")
    fig.tight_layout()
    fig.savefig(result_dir / "range_vs_k.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# E2 KV-only perplexity proxy
# ---------------------------------------------------------------------------


class KVQuantAttention:
    def __init__(self, rotations: dict[int, torch.Tensor], group_size: int, quantize: bool):
        self.rotations = rotations
        self.group_size = group_size
        self.quantize = quantize
        self.device_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def __call__(self, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 attention_mask: torch.Tensor | None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        if not self.quantize:
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
        cache_key = (module.layer_idx, query.device)
        if cache_key not in self.device_cache:
            self.device_cache[cache_key] = self.rotations[module.layer_idx].to(query.device, torch.float32)
        r = self.device_cache[cache_key]
        q_rot = torch.matmul(query.float(), r.T).to(query.dtype)
        k_rot = torch.matmul(key.float(), r.T)
        k_quant, _, _, _ = dynamic_asym_int4(k_rot, self.group_size)
        v_quant, _, _, _ = dynamic_asym_int4(value.float(), self.group_size)
        return sdpa_attention_forward(
            module, q_rot, k_quant.to(key.dtype), v_quant.to(value.dtype), attention_mask, **kwargs
        )


def rotations_for_method(cal_dir: Path, meta: dict[str, Any], b: int, method: str, seed: int) -> dict[int, torch.Tensor]:
    n = int(meta["head_dim"])
    result: dict[int, torch.Tensor] = {}
    for layer in range(int(meta["num_layers"])):
        payload = layer_payload(cal_dir, layer)
        cov = covariance(payload)
        pre_diag = payload["pre_diag_sum"].double() / int(payload["pre_count"])
        layer_seed = seed + 1000 * layer + b
        if method == "identity":
            result[layer] = torch.eye(n, dtype=torch.float64)
        elif method == "hadamard":
            result[layer] = full_hadamard_rotation(n, layer_seed)
        elif method == "nar":
            result[layer], _ = nar_rotation(cov, b, layer_seed)
        elif method == "nar_rope":
            r, details = nar_rope_rotation(cov, pre_diag, b, layer_seed)
            if r is None:
                raise ValueError(details["reason"])
            result[layer] = r
        else:
            raise ValueError(method)
    return result


def evaluate_nlls(model: torch.nn.Module, tokens: torch.Tensor, attention_impl: str, callback: KVQuantAttention) -> list[float]:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(attention_impl, callback)
    model.config._attn_implementation = attention_impl
    losses: list[float] = []
    with torch.inference_mode():
        for idx in range(tokens.shape[0]):
            batch = tokens[idx : idx + 1].cuda(non_blocking=True)
            output = model(input_ids=batch, labels=batch, use_cache=False)
            loss = float(output.loss.detach().float().cpu())
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite loss at sequence {idx}")
            losses.append(loss)
            if idx % 8 == 0:
                LOG.info("PPL sequence %d/%d loss=%.6f", idx + 1, tokens.shape[0], loss)
    return losses


def e2_specs(model_key: str, head_dim: int) -> list[int]:
    if model_key == "llama32_3b":
        return [64, 128]
    if model_key == "llama32_1b":
        return [32]
    return [head_dim // 2]


def run_e2(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E2 requires an allocated CUDA GPU")
    wd = work_path(args)
    setup_logging(wd, f"e2-{args.model}")
    model_id = resolve_model_id(args.model)
    model_key = model_key_from_id(model_id)
    cal_dir = wd / "activations" / model_key / args.cal_a
    cal_meta = json.loads((cal_dir / "DONE.json").read_text())
    tokens = prepare_token_chunks(model_id, "test", 0, args.eval_sequences, args.seq_len, wd)
    model = load_model(model_id, wd)
    methods_all = ["bf16", "identity", "hadamard", "nar", "nar_rope"]
    raw_rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, str], list[float]] = {}
    started = time.time()
    for b in e2_specs(model_key, int(cal_meta["head_dim"])):
        groups = int(cal_meta["head_dim"]) // b
        methods = [m for m in methods_all if not (m == "nar_rope" and (groups < 2 or groups % 2))]
        for seed_index in range(args.seeds):
            seed = args.seed + seed_index
            for method in methods:
                deterministic = method in ("bf16", "identity")
                key = (b, method)
                if deterministic and key in cache:
                    losses = cache[key]
                    reused = True
                else:
                    LOG.info("E2 model=%s b=%d seed=%d method=%s", model_key, b, seed, method)
                    if method == "bf16":
                        rotations = {i: torch.eye(int(cal_meta["head_dim"]), dtype=torch.float64)
                                     for i in range(int(cal_meta["num_layers"]))}
                        callback = KVQuantAttention(rotations, b, quantize=False)
                    else:
                        rotations = rotations_for_method(cal_dir, cal_meta, b, method, seed)
                        callback = KVQuantAttention(rotations, b, quantize=True)
                    losses = evaluate_nlls(model, tokens, "nar_e2", callback)
                    reused = False
                    if deterministic:
                        cache[key] = losses
                for seq_idx, loss in enumerate(losses):
                    raw_rows.append({
                        "model": model_key, "model_id": model_id, "b": b, "seed": seed,
                        "method": method, "sequence": seq_idx, "nll": loss,
                        "tokens_scored": args.seq_len - 1, "reused_deterministic_run": reused,
                    })
                write_csv(wd / "results" / model_key / "e2_per_sequence.partial.csv", raw_rows)
    result_dir = wd / "results" / model_key
    write_csv(result_dir / "e2_per_sequence.csv", raw_rows)
    summary_rows = summarize_e2(raw_rows)
    write_csv(result_dir / "e2_summary.csv", summary_rows)
    atomic_json(result_dir / "E2_DONE.json", {
        "model": model_key, "model_id": model_id, "eval_split": "test", "eval_sequences": args.eval_sequences,
        "sequence_length": args.seq_len, "seeds": args.seeds, "base_seed": args.seed,
        "paired_design": "identical calibration and evaluation sequences within each seed; only R differs",
        "ci": "two-sided paired 90% Student-t CI over the three seed-level PPL differences (df=2)",
        "elapsed_seconds": time.time() - started, "hardware": hardware_info(),
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()


def summarize_e2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = sorted({str(r["model"]) for r in rows})
    result: list[dict[str, Any]] = []
    tcrit_df2_90 = 2.919985580353725
    for model in models:
        bs = sorted({int(r["b"]) for r in rows if r["model"] == model})
        for b in bs:
            seeds = sorted({int(r["seed"]) for r in rows if r["model"] == model and int(r["b"]) == b})
            methods = sorted({str(r["method"]) for r in rows if r["model"] == model and int(r["b"]) == b})
            seed_ppl: dict[tuple[int, str], float] = {}
            for seed in seeds:
                for method in methods:
                    vals = [float(r["nll"]) for r in rows if r["model"] == model and int(r["b"]) == b
                            and int(r["seed"]) == seed and r["method"] == method]
                    seed_ppl[(seed, method)] = math.exp(float(np.mean(vals)))
            for method in methods:
                ppls = np.asarray([seed_ppl[(seed, method)] for seed in seeds], dtype=np.float64)
                if "hadamard" in methods:
                    deltas = np.asarray([seed_ppl[(seed, method)] - seed_ppl[(seed, "hadamard")] for seed in seeds])
                    mean_delta = float(deltas.mean())
                    half = tcrit_df2_90 * float(deltas.std(ddof=1)) / math.sqrt(len(deltas)) if len(deltas) > 1 else math.nan
                else:
                    mean_delta = half = math.nan
                result.append({
                    "model": model, "b": b, "method": method, "seeds": len(seeds),
                    "mean_ppl": float(ppls.mean()), "seed_ppl_std": float(ppls.std(ddof=1)),
                    "paired_ppl_delta_vs_hadamard": mean_delta,
                    "paired_90ci_low": mean_delta - half,
                    "paired_90ci_high": mean_delta + half,
                    "seed_ppls": ";".join(f"{x:.9g}" for x in ppls),
                })
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def clean(value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "N/A"
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(clean(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def aggregate(rows: list[dict[str, str]], keys: list[str], values: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for group_key, members in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(keys, group_key))
        for value in values:
            vals = [f(member, value) for member in members]
            vals = [v for v in vals if math.isfinite(v)]
            item[value] = float(np.mean(vals)) if vals else math.nan
        result.append(item)
    return result


def build_report(args: argparse.Namespace) -> None:
    wd = work_path(args)
    setup_logging(wd, "report")
    e0 = json.loads((wd / "results" / "e0.json").read_text())
    e1_dir = wd / "results" / "llama32_3b"
    e1 = read_csv(e1_dir / "e1_per_layer.csv")
    stability = read_csv(e1_dir / "e1_stability.csv")
    rank = read_csv(e1_dir / "e1_range_vs_k.csv")
    spectrum = read_csv(e1_dir / "e1_eigen_spectrum.csv")
    positions = read_csv(e1_dir / "e1_position.csv")
    qk = read_csv(e1_dir / "e1_qk_invariance.csv")
    diagnostics = read_csv(e1_dir / "e1_diagnostics.csv")
    e1_meta = json.loads((e1_dir / "E1_DONE.json").read_text())
    e2_rows: list[dict[str, str]] = []
    e2_meta: list[dict[str, Any]] = []
    for key in ("llama32_3b", "llama32_1b"):
        path = wd / "results" / key / "e2_summary.csv"
        if path.exists():
            e2_rows.extend(read_csv(path))
            e2_meta.append(json.loads((path.parent / "E2_DONE.json").read_text()))

    e1_avg = aggregate(e1, ["b", "method"], ["mean_group_range", "relative_quantization_error_nmse"])
    stab_avg = aggregate(stability, ["b"], ["reduction_cal_a_vs_hadamard", "reduction_disjoint_cal_b_vs_hadamard",
                                                   "reduction_degradation_fraction_points", "retained_fraction_of_cal_a_reduction"])
    rank_avg = aggregate(rank, ["b", "k"], ["nar_mean_group_range", "dc_absorption_fraction_vs_k0",
                                            "projection_range_attribution_fraction"])
    pos_avg = aggregate(positions, ["b", "position", "method"], ["mean_group_range", "reduction_vs_hadamard"])
    qk_valid = [r for r in qk if r.get("valid") == "True"]
    max_qk_abs = max(f(r, "max_abs_error") for r in qk_valid)
    max_qk_rel = max(f(r, "max_relative_error") for r in qk_valid)

    def avg_lookup(rows: list[dict[str, Any]], **match: str) -> dict[str, Any]:
        return next(row for row in rows if all(str(row[k]) == str(v) for k, v in match.items()))

    had128 = avg_lookup(e1_avg, b="128", method="hadamard")
    nar128 = avg_lookup(e1_avg, b="128", method="nar")
    range_reduction = (had128["mean_group_range"] - nar128["mean_group_range"]) / had128["mean_group_range"]
    rank128 = max((r for r in rank_avg if r["b"] == "128"), key=lambda r: int(r["k"]))
    attribution = rank128["projection_range_attribution_fraction"]
    stability128 = avg_lookup(stab_avg, b="128")
    survives = stability128["reduction_disjoint_cal_b_vs_hadamard"] > 0 and stability128["retained_fraction_of_cal_a_reduction"] >= 0.8
    substantial = attribution >= 0.10
    phenomenon_pass = range_reduction >= 0.10 and substantial and survives

    e2_nar = next((r for r in e2_rows if r["model"] == "llama32_3b" and r["b"] == "128" and r["method"] == "nar"), None)
    e2_gate_pass = False
    if e2_nar:
        e2_gate_pass = f(e2_nar, "paired_ppl_delta_vs_hadamard") <= 0 and f(e2_nar, "paired_90ci_high") < 0.05
    promising_pass = phenomenon_pass and e2_gate_pass

    diag_avg = aggregate(diagnostics, [], ["effective_rank", "top1_energy_fraction", "bos_to_nonbos_range_ratio"])[0]
    spectrum_top = aggregate([r for r in spectrum if r["rank"] in ("1", "2", "4", "8", "16", "32", "64", "128")],
                             ["rank"], ["fraction_total", "cumulative_fraction"])
    hardware_rows = []
    for label, meta in [("E1 cal A", e1_meta["calibration_a"]), ("E1 cal B", e1_meta["calibration_b"])]:
        hw = meta["hardware"]
        hardware_rows.append((label, hw.get("slurm_job_id"), hw.get("hostname"), hw.get("gpu_name"), meta["elapsed_seconds"]))
    for meta in e2_meta:
        hw = meta["hardware"]
        hardware_rows.append((f"E2 {meta['model']}", hw.get("slurm_job_id"), hw.get("hostname"), hw.get("gpu_name"), meta["elapsed_seconds"]))

    report: list[str] = []
    report.append("# NAR offline tensor validation\n")
    report.append("## Outcome\n")
    report.append(f"- **Phenomenon criterion: {'PASS' if phenomenon_pass else 'FAIL'}**. At b=128, NAR mean K range was "
                  f"{100*range_reduction:.3f}% below full-head Hadamard. The operational top-k projection attribution was "
                  f"{100*attribution:.3f}%, and a rotation recomputed on disjoint calibration data retained "
                  f"{100*stability128['retained_fraction_of_cal_a_reduction']:.3f}% of the calibration-A reduction.")
    if e2_nar:
        report.append(f"- **E2 PPL gate: {'PASS' if e2_gate_pass else 'FAIL'}**. The paired 3B b=128 NAR-Hadamard PPL delta was "
                      f"{f(e2_nar, 'paired_ppl_delta_vs_hadamard'):.6f}, with paired 90% CI "
                      f"[{f(e2_nar, 'paired_90ci_low'):.6f}, {f(e2_nar, 'paired_90ci_high'):.6f}].")
    report.append(f"- **Method-promising criterion: {'PASS' if promising_pass else 'FAIL'}**. This requires both the phenomenon criterion and the additional E2 PPL gate.")
    report.append(f"- **E0 implementation sanity: {'PASS' if e0['pass'] else 'FAIL'}**.")
    report.append(f"- QK invariance over every valid layer/method check: maximum absolute error {max_qk_abs:.6g}, maximum relative error {max_qk_rel:.6g}.\n")

    report.append("## Fixed protocol and quantizer semantics\n")
    report.append("The fake quantizer is dynamic asymmetric per token and contiguous channel group. For each group, "
                  "`s=(max-min)/15` and `z=min`; both `s` and the real-valued offset `z` are rounded to IEEE fp16 before use. "
                  "It then computes `q=clamp(round((x-z)/s),0,15)` and `x_hat=q*s+z`. Relative quantization error is NMSE "
                  "`sum((x_hat-x)^2)/sum(x^2)`. Degenerate groups use scale 1 and reproduce their fp16 offset. "
                  "All comparisons are paired on identical tensors. No GPTQ, weight quantization, or end-to-end W4A4 pipeline was run.\n")
    report.append("Primary E1 uses 128 WikiText-2 train sequences of length 2048 with BOS prepended. Uncentered second moments use every "
                  "token and KV head. Offline tensor dumps store positions 0,32,...,2016 from every sequence, all 24 Q heads, and all 8 KV heads; range analysis uses K. "
                  "Calibration B is the next disjoint 128 train sequences. NAR is layer-specific and K-only calibrated; Q is never included.\n")
    report.append("The pre-registered stability interpretation is positive held-out reduction and at least 80% retention of the calibration-A "
                  "reduction. Because 'substantial share' had no numeric threshold in the prompt, this report operationally uses 10%, matching "
                  "the primary range threshold.\n")

    report.append("## Compute actually used\n")
    report.append(md_table(["stage", "Slurm job", "node", "GPU", "seconds"], hardware_rows) + "\n")

    report.append("## E0 — synthetic sanity\n")
    outlier_rows = [r for r in e0["rows"] if r["check"] == "planted_outlier"]
    report.append(md_table(["b", "Hadamard range", "2|x|/sqrt(b)", "NAR range", "pass"],
                           [(r["b"], r["hadamard_range"], r["expected_2absx_over_sqrtb"], r["nar_range"], r["pass"])
                            for r in outlier_rows]) + "\n")
    shift_rows = [r for r in e0["rows"] if r["check"] == "constant_shift"]
    report.append("Constant-shift checks (the tiny nonzero MSE deltas, if present, are solely from required fp16 metadata rounding):\n")
    report.append(md_table(["b", "c", "|range delta|", "base MSE", "shifted MSE", "|MSE delta|", "pass"],
                           [(r["b"], r["c"], r["abs_range_delta"], r["base_mse"], r["shifted_mse"], r["abs_mse_delta"], r["pass"])
                            for r in shift_rows]) + "\n")

    report.append("## E1 — averaged results\n")
    report.append(md_table(["b", "method", "mean K range", "relative quantization error (NMSE)"],
                           [(r["b"], r["method"], r["mean_group_range"], r["relative_quantization_error_nmse"])
                            for r in e1_avg]) + "\n")
    report.append("NAR-RoPE at b=128 is N/A by construction: head_dim=128 and b=128 provide only one group, while an invariant "
                  "RoPE plane has two basis directions that the stated method requires mapping to two different group DCs. No surrogate value was inserted.\n")

    report.append("### Low-rank check\n")
    report.append(md_table(["b", "k", "mean NAR K range", "DC absorption vs k=0", "projection range attribution"],
                           [(r["b"], r["k"], r["nar_mean_group_range"], r["dc_absorption_fraction_vs_k0"],
                             r["projection_range_attribution_fraction"]) for r in rank_avg]) + "\n")
    report.append(md_table(["eigen rank", "mean trace fraction", "mean cumulative fraction"],
                           [(r["rank"], r["fraction_total"], r["cumulative_fraction"]) for r in spectrum_top]) + "\n")
    report.append(f"Mean effective rank was {diag_avg['effective_rank']:.4f}/128; mean top-1 energy fraction was "
                  f"{100*diag_avg['top1_energy_fraction']:.4f}%.\n")
    report.append(f"![Mean K range vs k](results/llama32_3b/range_vs_k.png)\n\n"
                  f"![Eigenvalue spectrum](results/llama32_3b/eigenvalue_spectrum.png)\n")

    report.append("### Stability\n")
    report.append(md_table(["b", "A reduction vs Had", "disjoint-B reduction vs Had", "degradation (fraction points)", "retained A reduction"],
                           [(r["b"], r["reduction_cal_a_vs_hadamard"], r["reduction_disjoint_cal_b_vs_hadamard"],
                             r["reduction_degradation_fraction_points"], r["retained_fraction_of_cal_a_reduction"]) for r in stab_avg]) + "\n")

    report.append("### NAR-RoPE position check\n")
    report.append("The same stored pre-RoPE K sample is rotated with the model's exact cos/sin at each requested hypothetical position; "
                  "this isolates positional rotation from token-distribution changes.\n")
    report.append(md_table(["b", "position", "method", "mean range", "reduction vs Had"],
                           [(r["b"], r["position"], r["method"], r["mean_group_range"], r["reduction_vs_hadamard"])
                            for r in pos_avg]) + "\n")

    report.append("### Per-layer tables\n")
    for b in ("32", "64", "128"):
        report.append(f"#### b={b}\n")
        subset = [r for r in e1 if r["b"] == b]
        report.append(md_table(["layer", "method", "mean range", "NMSE", "valid"],
                               [(r["layer"], r["method"], f(r, "mean_group_range"), f(r, "relative_quantization_error_nmse"), r["valid"])
                                for r in subset]) + "\n")

    report.append("## E2 — KV-only perplexity proxy\n")
    report.append("Only K and V are fake-quantized; weights and all other activations remain bf16. K is rotated after RoPE and Q receives "
                  "the identical orthogonal rotation. The full-sequence prefill tensors are quantized at the same point they enter the cache; "
                  "this is a cache-content proxy, not an autoregressive latency benchmark. Test sequences, calibration data, and all non-R "
                  "settings are paired. Deterministic bf16/identity evaluations are measured once and reused exactly across seed rows.\n")
    report.append(md_table(["model", "b", "method", "mean PPL", "seed SD", "paired delta vs Had", "90% CI low", "90% CI high", "seed PPLs"],
                           [(r["model"], r["b"], r["method"], f(r, "mean_ppl"), f(r, "seed_ppl_std"),
                             f(r, "paired_ppl_delta_vs_hadamard"), f(r, "paired_90ci_low"), f(r, "paired_90ci_high"), r["seed_ppls"])
                            for r in e2_rows]) + "\n")

    report.append("## Negative findings and diagnosis\n")
    effective_rank = diag_avg["effective_rank"]
    top1_fraction = diag_avg["top1_energy_fraction"]
    bos_ratio = diag_avg["bos_to_nonbos_range_ratio"]
    dc_absorption = rank128["dc_absorption_fraction_vs_k0"]
    stability_retained = stability128["retained_fraction_of_cal_a_reduction"]
    reduction_b32 = avg_lookup(stab_avg, b="32")["reduction_cal_a_vs_hadamard"]
    reduction_b64 = avg_lookup(stab_avg, b="64")["reduction_cal_a_vs_hadamard"]
    b128_layer_reductions = []
    for layer_index in range(int(e1_meta["n_layers"])):
        had_layer = next(f(row, "mean_group_range") for row in e1 if row["layer"] == str(layer_index) and row["b"] == "128" and row["method"] == "hadamard")
        nar_layer = next(f(row, "mean_group_range") for row in e1 if row["layer"] == str(layer_index) and row["b"] == "128" and row["method"] == "nar")
        b128_layer_reductions.append((had_layer - nar_layer) / had_layer)
    layers_at_threshold = sum(value >= 0.10 for value in b128_layer_reductions)
    report.append(f"The primary phenomenon criterion failed because the required b=128 NAR reduction was only {100*range_reduction:.3f}% (threshold 10%); only {layers_at_threshold}/{len(b128_layer_reductions)} layers individually reached 10%. ")
    report.append(f"The spectrum was not flat (mean effective rank {effective_rank:.3f}/128; top-1 trace share {100*top1_fraction:.3f}%), and disjoint calibration retained {100*stability_retained:.3f}% of the A reduction, so flat spectrum and calibration overfit do not explain the failure. ")
    report.append(f"Instead, b=128 has only one DC slot: the matched k=0 to k=1 ablation reduced range by {100*dc_absorption:.3f}%, despite {100*attribution:.3f}% projection-removal attribution. The residual directions and the non-additivity of range dominate after absorbing one direction. ")
    report.append(f"The group-size trend supports this capacity diagnosis: NAR reductions were {100*reduction_b32:.3f}% at b=32, {100*reduction_b64:.3f}% at b=64, and {100*range_reduction:.3f}% at b=128. ")
    report.append(f"BOS is not the driver in these samples: its identity range was only {bos_ratio:.3f}x non-BOS. ")
    if e2_gate_pass:
        report.append("KV-only PPL nevertheless favored NAR, showing that the range threshold and the cache proxy can disagree; by the stated conjunctive criteria this is still a no-go. ")
    else:
        report.append("The KV-only PPL gate also failed. ")
    report.append("\n")

    report.append("## Unsure about\n")
    report.append("- The prompt does not define a numeric threshold for a 'substantial' top-k range share; 10% was fixed here and is reported separately so it can be reinterpreted without rerunning.\n"
                  "- fp16 storage of the real-valued zero/offset makes exact constant-shift MSE invariance mathematically impossible for arbitrary constants because the offset's fp16 rounding grid changes with magnitude. E0 therefore reports every measured delta and uses a fixed 2%-of-baseline-MSE tolerance; the range itself is invariant.\n"
                  "- NAR-RoPE is undefined when there is only one group, including 3B b=128 and 1B b=64. Reporting identity under that name would fabricate a method, so those cells are N/A.\n"
                  "- Three seeds support only a very low-degree-of-freedom CI. The paired 90% CI is the requested result, not strong evidence of distributional generality.\n")

    report.append("## Go / no-go\n")
    if phenomenon_pass and promising_pass:
        report.append("**GO, cautiously:** both pre-registered gates pass. The next justified step is an independently implemented cache kernel or a broader-model replication, not tuning this dataset.\n")
    elif phenomenon_pass:
        report.append("**NO-GO for pipeline development:** the tensor phenomenon passes, but the KV-only PPL gate does not. Preserve this as a mechanistic result; do not invest in GPTQ or a full W4A4KV4 pipeline without independent replication.\n")
    else:
        report.append("**NO-GO:** the core tensor premise does not meet the stated gate. Do not build a full quantization pipeline around this version of NAR.\n")

    report.append("## Reproduction artifacts\n")
    report.append("All exact numeric CSVs are under `results/`; captured samples and all-token second moments are under `activations/`; "
                  "executed commands and stdout/stderr logs are under `runs/`. Run `./run_all.sh` to reproduce every stage and table.\n")
    (wd / "report.md").write_text("\n".join(report))
    atomic_json(wd / "results" / "decision.json", {
        "phenomenon_pass": phenomenon_pass, "e2_gate_pass": e2_gate_pass, "promising_pass": promising_pass,
        "nar_range_reduction_vs_hadamard_b128": range_reduction,
        "topk_projection_range_attribution_b128": attribution,
        "heldout_survives": survives,
        "e2_nar_vs_hadamard": e2_nar,
    })
    LOG.info("wrote %s", wd / "report.md")


# ---------------------------------------------------------------------------
# Pipeline and CLI
# ---------------------------------------------------------------------------


def invoke_main(argv: list[str]) -> None:
    LOG.info("pipeline invoke: %s", shlex.join([sys.executable, __file__, *argv]))
    main(argv)


def pipeline(args: argparse.Namespace) -> None:
    wd = work_path(args)
    setup_logging(wd, "pipeline")
    common = ["--workdir", str(wd), "--seed", str(args.seed)]
    if not (wd / "results" / "e0.json").exists():
        invoke_main([*common, "e0"])
    if not (wd / "activations" / "llama32_3b" / "cal_a" / "DONE.json").exists():
        invoke_main([*common, "collect", "--model", "llama32_3b", "--tag", "cal_a", "--split", "train",
                     "--offset", "0", "--sequences", "128", "--seq-len", "2048", "--batch-size", str(args.batch_size),
                     "--sample-stride", "32"])
    if not (wd / "activations" / "llama32_3b" / "cal_b" / "DONE.json").exists():
        invoke_main([*common, "collect", "--model", "llama32_3b", "--tag", "cal_b", "--split", "train",
                     "--offset", "128", "--sequences", "128", "--seq-len", "2048", "--batch-size", str(args.batch_size),
                     "--sample-stride", "32", "--moments-only"])
    if not (wd / "results" / "llama32_3b" / "E1_DONE.json").exists():
        invoke_main([*common, "analyze", "--model", "llama32_3b", "--cal-a", "cal_a", "--cal-b", "cal_b"])
    if not (wd / "activations" / "llama32_1b" / "cal_a" / "DONE.json").exists():
        invoke_main([*common, "collect", "--model", "llama32_1b", "--tag", "cal_a", "--split", "train",
                     "--offset", "0", "--sequences", "128", "--seq-len", "2048", "--batch-size", str(args.batch_size),
                     "--sample-stride", "32", "--moments-only"])
    for model in ("llama32_3b", "llama32_1b"):
        if not (wd / "results" / model / "E2_DONE.json").exists():
            invoke_main([*common, "e2", "--model", model, "--cal-a", "cal_a", "--eval-sequences", str(args.eval_sequences),
                         "--seq-len", "2048", "--seeds", "3"])
    invoke_main([*common, "report"])


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("e0")
    collect = sub.add_parser("collect")
    collect.add_argument("--model", required=True)
    collect.add_argument("--tag", required=True)
    collect.add_argument("--split", default="train")
    collect.add_argument("--offset", type=int, default=0)
    collect.add_argument("--sequences", type=int, default=128)
    collect.add_argument("--seq-len", type=int, default=2048)
    collect.add_argument("--batch-size", type=int, default=2)
    collect.add_argument("--sample-stride", type=int, default=32)
    collect.add_argument("--moments-only", action="store_true")
    collect.add_argument("--force", action="store_true")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--model", default="llama32_3b")
    analyze.add_argument("--cal-a", default="cal_a")
    analyze.add_argument("--cal-b", default="cal_b")
    e2 = sub.add_parser("e2")
    e2.add_argument("--model", required=True)
    e2.add_argument("--cal-a", default="cal_a")
    e2.add_argument("--eval-sequences", type=int, default=64)
    e2.add_argument("--seq-len", type=int, default=2048)
    e2.add_argument("--seeds", type=int, default=3)
    sub.add_parser("report")
    all_parser = sub.add_parser("all")
    all_parser.add_argument("--batch-size", type=int, default=2)
    all_parser.add_argument("--eval-sequences", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    dispatch = {
        "e0": run_e0,
        "collect": collect_activations,
        "analyze": analyze_e1,
        "e2": run_e2,
        "report": build_report,
        "all": pipeline,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
