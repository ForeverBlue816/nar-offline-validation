#!/usr/bin/env python3
"""E15 FP4-E2M1 boundary check on the frozen E1c activation dumps."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from . import extended_experiment as ext
except ImportError:
    import activation_experiments as act
    import experiment as base
    import extended_experiment as ext


LOG = logging.getLogger("nar")
MODEL = "llama32_3b"
SITES = ("q_input", "down_input")
BLOCK = 16
METHODS = ("bf16", "identity", "hadamard", "nar", "logdiag")
E2M1 = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6])


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def e2m1_e4m3_qdq(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest E2M1 with one E4M3FN max-calibrated scale per 16 values."""
    blocks = value.float().reshape(-1, BLOCK)
    maximum = blocks.abs().amax(-1, keepdim=True)
    raw = maximum / 6
    safe = raw.clamp(min=2 ** -9, max=448).to(torch.float8_e4m3fn).float()
    safe = torch.where(maximum > 0, safe, torch.ones_like(safe))
    normalized = blocks / safe
    codebook = E2M1.to(blocks.device)
    indices = (normalized.unsqueeze(-1) - codebook).abs().argmin(-1)
    output = codebook[indices] * safe
    relative_scale_error = torch.where(raw > 0, (safe - raw).abs() / raw, torch.zeros_like(raw))
    return output.reshape_as(value).to(value.dtype), relative_scale_error


def pearson_kurtosis(value: torch.Tensor) -> float:
    rows = value.float().reshape(-1)
    centered = rows - rows.mean()
    variance = centered.square().mean()
    return float(centered.pow(4).mean() / variance.square().clamp_min(1e-30))


def _logdiag(n: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    diagonal = torch.exp(torch.linspace(-math.log(4), math.log(4), n))
    return diagonal[torch.randperm(n, generator=generator)].to(device)


def _block_hadamard(value: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Random-sign orthonormal H16 applied inside each FP4 scale block."""
    shape = value.shape
    blocks = (value.float() * signs).reshape(-1, shape[-1] // BLOCK, BLOCK)
    return ext._fast_walsh_hadamard(blocks).reshape(shape)


def analyze(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E15 requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e15-fp4")
    result_dir = workdir / "results" / MODEL
    done = result_dir / "E15_DONE.json"
    if done.exists():
        LOG.info("E15 exists: %s", done)
        return
    wide = workdir / "activations" / MODEL / "wide_cal_a"
    meta = _json(wide / "DONE.json")
    dimensions = {"qkv": int(meta["hidden_size"]), "down": int(meta["intermediate_size"])}
    rotations = act.MethodRotations(
        workdir, MODEL, "nar", 0, args.seed,
        int(meta["num_layers"]), dimensions, torch.device("cuda")
    )
    rows = []
    for site in SITES:
        rotation_site = "qkv" if site == "q_input" else "down"
        for layer in range(int(meta["num_layers"])):
            mmap = ext._open_site(wide, meta, site, layer)
            value = ext._sample_site_tokens(mmap, args.sample_stride, torch.device("cuda"))
            base_kurtosis = pearson_kurtosis(value)
            signal_energy = float(value.float().square().sum(dtype=torch.float64))
            rows.append({
                "model": MODEL, "site": site, "layer": layer, "method": "bf16",
                "fp4_nmse": 0.0, "transformed_pearson_kurtosis": base_kurtosis,
                "mean_relative_e4m3_scale_rounding_error": 0.0,
                "squared_error": 0.0, "signal_energy": signal_energy,
                "sample_vectors": value.shape[0],
            })
            for method in METHODS[1:]:
                diagonal = None
                if method == "identity":
                    transformed = value.float()
                elif method == "hadamard":
                    transformed = _block_hadamard(
                        value, rotations.signs[(rotation_site, layer)]
                    )
                elif method == "nar":
                    transformed = rotations.apply(rotation_site, layer, value)
                else:
                    diagonal = _logdiag(value.shape[-1], args.seed + 10_000 * (site == "down_input") + layer,
                                        value.device)
                    transformed = value.float() * diagonal
                quantized, scale_error = e2m1_e4m3_qdq(transformed)
                if diagonal is None:
                    numerator = (quantized.float() - transformed.float()).square().sum(
                        dtype=torch.float64
                    )
                else:
                    reconstructed = quantized.float() / diagonal
                    numerator = (reconstructed - value.float()).square().sum(dtype=torch.float64)
                nmse = float(numerator / max(signal_energy, 1e-30))
                rows.append({
                    "model": MODEL, "site": site, "layer": layer, "method": method,
                    "fp4_nmse": nmse,
                    "transformed_pearson_kurtosis": pearson_kurtosis(transformed),
                    "mean_relative_e4m3_scale_rounding_error": float(scale_error.mean()),
                    "squared_error": float(numerator), "signal_energy": signal_energy,
                    "sample_vectors": value.shape[0],
                })
                del transformed, quantized, scale_error
            LOG.info("E15 %s layer %d/%d", site, layer + 1, meta["num_layers"])
            del mmap, value
            torch.cuda.empty_cache()
    base.write_csv(result_dir / "e15_fp4_per_layer.csv", rows)
    summary = []
    for site in SITES:
        for method in METHODS:
            subset = [row for row in rows if row["site"] == site and row["method"] == method]
            squared_error = sum(row["squared_error"] for row in subset)
            signal_energy = sum(row["signal_energy"] for row in subset)
            summary.append({
                "model": MODEL, "site": site, "method": method, "layers": len(subset),
                "global_fp4_nmse": squared_error / max(signal_energy, 1e-30),
                "mean_layer_fp4_nmse": float(np.mean([row["fp4_nmse"] for row in subset])),
                "mean_transformed_pearson_kurtosis": float(np.mean([
                    row["transformed_pearson_kurtosis"] for row in subset
                ])),
                "mean_relative_e4m3_scale_rounding_error": float(np.mean([
                    row["mean_relative_e4m3_scale_rounding_error"] for row in subset
                ])),
            })
    base.write_csv(result_dir / "e15_fp4_summary.csv", summary)
    comparisons = []
    for site in SITES:
        had_rows = [row for row in rows if row["site"] == site and row["method"] == "hadamard"]
        nar_rows = [row for row in rows if row["site"] == site and row["method"] == "nar"]
        had = np.asarray([row["fp4_nmse"] for row in had_rows])
        nar = np.asarray([row["fp4_nmse"] for row in nar_rows])
        delta = nar - had
        global_had = sum(row["squared_error"] for row in had_rows) / sum(
            row["signal_energy"] for row in had_rows
        )
        global_nar = sum(row["squared_error"] for row in nar_rows) / sum(
            row["signal_energy"] for row in nar_rows
        )
        comparisons.append({
            "model": MODEL, "site": site, "mean_nar_minus_hadamard_nmse": float(delta.mean()),
            "global_nar_minus_hadamard_nmse": global_nar - global_had,
            "nar_better_layers": int((delta < 0).sum()), "layers": len(delta),
            "nar_loses_zero_point_advantage": bool(global_nar >= global_had),
        })
    base.write_csv(result_dir / "e15_fp4_comparison.csv", comparisons)
    base.atomic_json(done, {
        "model": MODEL, "source": str(wide), "sample_stride": args.sample_stride,
        "paired": "identical frozen E1c vectors for all methods",
        "fp4": "nearest finite E2M1; one max/6 E4M3FN scale per block of 16; no zero-point",
        "hadamard": "fixed random-sign orthonormal H16 inside each aligned FP4 scale block",
        "logdiag": "fixed seeded diagonal exp(linspace(-ln4,ln4)) with random permutation; condition number 16; exact inverse applied before NMSE",
        "comparison": comparisons, "seed": args.seed, "no_tuning": True,
        "hardware": base.hardware_info(),
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--sample-stride", type=int, default=128)
    return parser


if __name__ == "__main__":
    analyze(parser().parse_args())
