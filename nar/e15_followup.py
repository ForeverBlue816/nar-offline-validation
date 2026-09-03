#!/usr/bin/env python3
"""Paired verification of the unexpected E15 FP4 result.

The original E15 files remain immutable.  This follow-up compares absmax and
exact E4M3FN-scale MSE selection, reports block-error tails, and isolates the
NAR-group/FP4-block mismatch with a matched b=16 NAR construction.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from . import extended_experiment as ext
    from .e15_fp4 import BLOCK, E2M1, MODEL, pearson_kurtosis
except ImportError:
    import activation_experiments as act
    import experiment as base
    import extended_experiment as ext
    from e15_fp4 import BLOCK, E2M1, MODEL, pearson_kurtosis


SITES = ("q_input", "down_input")
METHODS = ("identity", "hadamard_b16", "nar_b128", "nar_b16")
SCALE_RULES = ("absmax", "mse_optimal")
CALIBRATION_STRIDE = 32


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def e4m3fn_positive_scales(device: torch.device) -> torch.Tensor:
    """All 126 positive finite E4M3FN values, exactly once and sorted."""
    bits = torch.arange(1, 127, dtype=torch.uint8)
    scales = bits.view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(scales).all() and (scales > 0).all()):
        raise AssertionError("invalid E4M3FN scale enumeration")
    return scales.to(device)


def _nearest_e2m1(normalized: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    boundaries = (levels[:-1] + levels[1:]) / 2
    indices = torch.bucketize(normalized, boundaries)
    return levels[indices]


def block_errors_absmax(blocks: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    maximum = blocks.abs().amax(-1, keepdim=True)
    raw_scale = maximum / 6.0
    scale = raw_scale.clamp(min=2 ** -9, max=448).to(torch.float8_e4m3fn).float()
    scale = torch.where(maximum > 0, scale, torch.ones_like(scale))
    reconstructed = _nearest_e2m1(blocks / scale, levels) * scale
    return (reconstructed - blocks).square().sum(-1)


def block_errors_mse_optimal(
    blocks: torch.Tensor,
    levels: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Exact argmin over every positive finite E4M3FN block scale."""
    best = torch.full((blocks.shape[0],), torch.inf, device=blocks.device)
    # A scalar-scale loop bounds memory while still exhaustively evaluating
    # the complete discrete scale set; no clipping grid or tuned window.
    for scale in scales:
        reconstructed = _nearest_e2m1(blocks / scale, levels) * scale
        error = (reconstructed - blocks).square().sum(-1)
        best = torch.minimum(best, error)
    return best


def _random_signs(n: int, layer: int, site: str, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        act._seed(seed, 0, layer, "qkv" if site == "q_input" else "down")
    )
    return torch.randint(0, 2, (n,), generator=generator).float().mul_(2).sub_(1).to(device)


def b16_factor_dir(workdir: Path) -> Path:
    return workdir / "activations" / MODEL / "e15_nar_b16_factors"


def build_b16_factors(workdir: Path, device: torch.device) -> None:
    output = b16_factor_dir(workdir)
    done = output / "DONE.json"
    if done.exists():
        return
    wide = workdir / "activations" / MODEL / "wide_cal_a"
    meta = _json(wide / "DONE.json")
    eig_dir = wide / "analysis" / "eigenspaces"
    output.mkdir(parents=True, exist_ok=True)
    errors: list[float] = []
    ranks: dict[str, int] = {}
    for site in SITES:
        target_site = "qkv" if site == "q_input" else "down"
        for layer in range(int(meta["num_layers"])):
            path = output / f"{target_site}_layer_{layer:02d}.pt"
            if path.exists():
                continue
            eig = torch.load(
                eig_dir / f"{site}_layer_{layer:02d}.pt",
                map_location="cpu", weights_only=True,
            )
            vectors = eig["vectors"].float().to(device)
            ranks[site] = int(vectors.shape[1])
            mmap = ext._open_site(wide, meta, site, layer)
            calibration = ext._sample_site_tokens(mmap, CALIBRATION_STRIDE, device)
            factor = act.factor_from_vectors(vectors, calibration, BLOCK)
            factor.save(path, {
                "source": "frozen E1c eigendirections and calibration tokens",
                "site": target_site,
                "layer": layer,
                "group_size": BLOCK,
                "rank_policy": "same top-direction count as preregistered NAR-b128",
            })
            errors.append(factor.anchor_error)
            del eig, vectors, mmap, calibration, factor
            gc.collect()
            torch.cuda.empty_cache()
    base.atomic_json(done, {
        "model": MODEL,
        "group_size": BLOCK,
        "fp4_block_size": BLOCK,
        "ranks": ranks,
        "rank_policy": "same frozen V and k as NAR-b128; only group/DC spacing changes",
        "calibration_stride": CALIBRATION_STRIDE,
        "max_anchor_error": max(errors) if errors else 0.0,
        "no_tuning": True,
    })


def transform_rows(
    method: str,
    value: torch.Tensor,
    signs: torch.Tensor,
    b128: act.RotationFactor,
    b16: act.RotationFactor,
) -> torch.Tensor:
    if method == "identity":
        return value.float()
    if method == "hadamard_b16":
        shape = value.shape
        blocks = (value.float() * signs).reshape(-1, shape[-1] // BLOCK, BLOCK)
        return ext._fast_walsh_hadamard(blocks).reshape(shape)
    if method == "nar_b128":
        return b128.apply(value, signs)
    if method == "nar_b16":
        return b16.apply(value, signs)
    raise ValueError(method)


def _pearson(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2 or xa.std() == 0 or ya.std() == 0:
        return math.nan
    return float(np.corrcoef(xa, ya)[0, 1])


class TailAccumulator:
    def __init__(self, total_blocks: int, device: torch.device):
        self.total_blocks = total_blocks
        self.keep = max(1, math.ceil(0.01 * total_blocks))
        self.device = device
        self.nmse: list[np.ndarray] = []
        self.top_error = torch.empty(0, device=device)
        self.top_signal = torch.empty(0, device=device)
        self.error_sum = 0.0
        self.signal_sum = 0.0
        self.count = 0

    def add(self, error: torch.Tensor, signal: torch.Tensor) -> None:
        nmse = torch.where(signal > 0, error / signal, torch.zeros_like(error))
        self.nmse.append(nmse.float().cpu().numpy())
        self.error_sum += float(error.double().sum())
        self.signal_sum += float(signal.double().sum())
        self.count += error.numel()
        errors = torch.cat((self.top_error, error.float()))
        signals = torch.cat((self.top_signal, signal.float()))
        keep = min(self.keep, errors.numel())
        indices = torch.topk(errors, keep, sorted=False).indices
        self.top_error = errors[indices]
        self.top_signal = signals[indices]

    def summarize(self) -> dict[str, float | int]:
        if self.count != self.total_blocks:
            raise AssertionError(f"block count {self.count} != {self.total_blocks}")
        values = np.concatenate(self.nmse)
        result: dict[str, float | int] = {
            "blocks": self.count,
            "global_nmse": self.error_sum / max(self.signal_sum, 1e-30),
            "block_nmse_median": float(np.quantile(values, 0.50)),
            "block_nmse_p90": float(np.quantile(values, 0.90)),
            "block_nmse_p99": float(np.quantile(values, 0.99)),
            "worst_1pct_error_share": float(self.top_error.double().sum()) / max(self.error_sum, 1e-30),
            "worst_1pct_signal_energy_share": float(self.top_signal.double().sum()) / max(self.signal_sum, 1e-30),
        }
        return result


def analyze(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E15 follow-up requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e15-followup")
    result_dir = workdir / "results" / MODEL
    done = result_dir / "E15_FOLLOWUP_DONE.json"
    if done.exists():
        return
    device = torch.device("cuda")
    build_b16_factors(workdir, device)
    wide = workdir / "activations" / MODEL / "wide_cal_a"
    meta = _json(wide / "DONE.json")
    layers = int(meta["num_layers"])
    levels = E2M1.to(device)
    e4m3_scales = e4m3fn_positive_scales(device)
    b128_rotations = act.MethodRotations(
        workdir, MODEL, "nar", 0, args.seed, layers,
        {"qkv": int(meta["hidden_size"]), "down": int(meta["intermediate_size"])},
        device,
    )
    layer_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for site in SITES:
        n = int(meta["hidden_size"] if site == "q_input" else meta["intermediate_size"])
        target_site = "qkv" if site == "q_input" else "down"
        rows_per_layer = meta["site_shapes"][site][0] * math.ceil(meta["site_shapes"][site][1] / args.sample_stride)
        total_blocks = int(rows_per_layer * (n // BLOCK) * layers)
        accumulators = {
            (method, rule): TailAccumulator(total_blocks, device)
            for method in METHODS for rule in SCALE_RULES
        }
        for layer in range(layers):
            mmap = ext._open_site(wide, meta, site, layer)
            value = ext._sample_site_tokens(mmap, args.sample_stride, device)
            signs = _random_signs(n, layer, site, args.seed, device)
            b128 = b128_rotations.factors[(target_site, layer)]
            b16 = act.RotationFactor.load(
                b16_factor_dir(workdir) / f"{target_site}_layer_{layer:02d}.pt", device
            )
            for method in METHODS:
                transformed = transform_rows(method, value, signs, b128, b16)
                blocks = transformed.reshape(-1, BLOCK)
                signal = blocks.square().sum(-1)
                kurtosis = pearson_kurtosis(transformed)
                abs_error = block_errors_absmax(blocks, levels)
                mse_error = block_errors_mse_optimal(blocks, levels, e4m3_scales)
                for rule, error in (("absmax", abs_error), ("mse_optimal", mse_error)):
                    accumulators[(method, rule)].add(error, signal)
                    layer_rows.append({
                        "model": MODEL,
                        "site": site,
                        "layer": layer,
                        "method": method,
                        "nar_group_size": 128 if method == "nar_b128" else 16 if method == "nar_b16" else "",
                        "fp4_block_size": BLOCK,
                        "scale_rule": rule,
                        "layer_global_nmse": float(error.double().sum() / signal.double().sum().clamp_min(1e-30)),
                        "transformed_pearson_kurtosis": kurtosis,
                        "blocks": error.numel(),
                    })
                del transformed, blocks, signal, abs_error, mse_error
            del mmap, value, b16
            gc.collect()
            torch.cuda.empty_cache()

        for (method, rule), accumulator in accumulators.items():
            distribution_rows.append({
                "model": MODEL,
                "site": site,
                "method": method,
                "nar_group_size": 128 if method == "nar_b128" else 16 if method == "nar_b16" else "",
                "fp4_block_size": BLOCK,
                "scale_rule": rule,
                **accumulator.summarize(),
            })
        del accumulators
        gc.collect()

    correlation_rows: list[dict[str, Any]] = []
    for site in SITES:
        for method in METHODS:
            kurtosis = [
                float(row["transformed_pearson_kurtosis"]) for row in layer_rows
                if row["site"] == site and row["method"] == method and row["scale_rule"] == "absmax"
            ]
            for rule in SCALE_RULES:
                nmse = [
                    float(row["layer_global_nmse"]) for row in layer_rows
                    if row["site"] == site and row["method"] == method and row["scale_rule"] == rule
                ]
                correlation_rows.append({
                    "model": MODEL, "site": site, "method": method, "scale_rule": rule,
                    "pearson_layer_kurtosis_vs_nmse": _pearson(kurtosis, nmse), "layers": layers,
                })
        had_k = {
            int(row["layer"]): float(row["transformed_pearson_kurtosis"])
            for row in layer_rows if row["site"] == site and row["method"] == "hadamard_b16"
        }
        for method in ("nar_b128", "nar_b16"):
            nar_k = {
                int(row["layer"]): float(row["transformed_pearson_kurtosis"])
                for row in layer_rows if row["site"] == site and row["method"] == method
            }
            delta_k = [nar_k[layer] - had_k[layer] for layer in range(layers)]
            for rule in SCALE_RULES:
                by_method = {
                    (str(row["method"]), int(row["layer"])): float(row["layer_global_nmse"])
                    for row in layer_rows if row["site"] == site and row["scale_rule"] == rule
                }
                delta_nmse = [
                    by_method[(method, layer)] - by_method[("hadamard_b16", layer)]
                    for layer in range(layers)
                ]
                correlation_rows.append({
                    "model": MODEL, "site": site,
                    "method": f"{method}_minus_hadamard_b16", "scale_rule": rule,
                    "pearson_layer_kurtosis_vs_nmse": _pearson(delta_k, delta_nmse), "layers": layers,
                })

    def summary(site: str, method: str, rule: str) -> dict[str, Any]:
        return next(
            row for row in distribution_rows
            if row["site"] == site and row["method"] == method and row["scale_rule"] == rule
        )

    identity_diagnosis = {}
    for rule in SCALE_RULES:
        identity = float(summary("q_input", "identity", rule)["global_nmse"])
        hadamard = float(summary("q_input", "hadamard_b16", rule)["global_nmse"])
        identity_diagnosis[rule] = {
            "identity_nmse": identity,
            "hadamard_nmse": hadamard,
            "identity_better": identity < hadamard,
            "identity_minus_hadamard": identity - hadamard,
        }
    scale_artifact = not identity_diagnosis["mse_optimal"]["identity_better"]

    base.write_csv(result_dir / "e15_followup_per_layer.csv", layer_rows)
    base.write_csv(result_dir / "e15_followup_block_distribution.csv", distribution_rows)
    base.write_csv(result_dir / "e15_followup_kurtosis_correlation.csv", correlation_rows)
    base.atomic_json(done, {
        "model": MODEL,
        "paired": "identical frozen E1c token vectors for every method and scale rule",
        "sample_stride": args.sample_stride,
        "e2m1_block_size": BLOCK,
        "scale_rules": {
            "absmax": "round clamp(max(abs(x))/6, 2^-9, 448) to nearest E4M3FN",
            "mse_optimal": "exact exhaustive argmin of block SSE over all 126 positive finite E4M3FN scale codes",
        },
        "nar_b128": "preregistered factor: k=24 q_input, k=64 down_input; DC spacing and terminal Hadamard b=128",
        "nar_b16": "same frozen directions and same k; DC spacing and terminal Hadamard changed to b=16 to match FP4 blocks",
        "tail_definition": "worst 1% selected globally by per-block squared error; signal share is for the same selected blocks",
        "identity_q_input": identity_diagnosis,
        "identity_best_is_absmax_scale_artifact": scale_artifact,
        "no_tuning": True,
        "negative_results_reported": True,
        "hardware": base.hardware_info(),
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--sample-stride", type=int, default=128)
    return result


if __name__ == "__main__":
    analyze(parser().parse_args())
