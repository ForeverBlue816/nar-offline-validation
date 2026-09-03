#!/usr/bin/env python3
"""E15 final control: frozen NAR permutation without directional alignment."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from . import extended_experiment as ext
    from .e15_fp4 import BLOCK, E2M1, MODEL
    from .e15_followup import (
        SCALE_RULES,
        SITES,
        TailAccumulator,
        _random_signs,
        block_errors_absmax,
        block_errors_mse_optimal,
        e4m3fn_positive_scales,
        transform_rows,
    )
    from .e15_h128_control import h128_rows
except ImportError:
    import activation_experiments as act
    import experiment as base
    import extended_experiment as ext
    from e15_fp4 import BLOCK, E2M1, MODEL
    from e15_followup import (
        SCALE_RULES,
        SITES,
        TailAccumulator,
        _random_signs,
        block_errors_absmax,
        block_errors_mse_optimal,
        e4m3fn_positive_scales,
        transform_rows,
    )
    from e15_h128_control import h128_rows


MIXING_WIDTH = 128
METHODS = ("hadamard_b128", "hadamard_b128_pi", "nar_b128", "hadamard_b16")
RECOVERY_LOW = 0.30
RECOVERY_HIGH = 0.70


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pi_h128_rows(
    value: torch.Tensor,
    signs: torch.Tensor,
    factor: act.RotationFactor,
) -> torch.Tensor:
    """Apply frozen Pi, signs, H128; deliberately omit every reflector in G."""
    shape = value.shape
    rows = value.float().reshape(-1, factor.n)
    permuted = torch.empty_like(rows)
    permuted[:, factor.target_order] = rows[:, factor.source_order]
    blocks = (permuted * signs).reshape(-1, factor.n // MIXING_WIDTH, MIXING_WIDTH)
    return ext._fast_walsh_hadamard(blocks).reshape(shape)


def mean_group_energy_cv(value: torch.Tensor) -> tuple[float, int]:
    """Mean over tokens of CV across per-128-group signal energies."""
    groups = value.float().reshape(-1, value.shape[-1] // MIXING_WIDTH, MIXING_WIDTH)
    energy = groups.square().sum(-1)
    means = energy.mean(-1)
    cv = energy.std(-1, unbiased=False) / means.clamp_min(1e-30)
    return float(cv.double().sum()), cv.numel()


def _old_row(
    followup: list[dict[str, str]],
    h128: list[dict[str, str]],
    site: str,
    method: str,
    rule: str,
) -> dict[str, str]:
    source = h128 if method == "hadamard_b128" else followup
    source_method = {
        "hadamard_b128": "hadamard_b128",
        "nar_b128": "nar_b128",
        "hadamard_b16": "hadamard_b16",
    }[method]
    return next(
        row for row in source
        if row["site"] == site and row["method"] == source_method and row["scale_rule"] == rule
    )


def analyze(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E15 Pi control requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e15-pi-control")
    result_dir = workdir / "results" / MODEL
    done = result_dir / "E15_PI_CONTROL_DONE.json"
    if done.exists():
        return
    wide = workdir / "activations" / MODEL / "wide_cal_a"
    meta = _json(wide / "DONE.json")
    layers = int(meta["num_layers"])
    dimensions = {
        "qkv": int(meta["hidden_size"]),
        "down": int(meta["intermediate_size"]),
    }
    device = torch.device("cuda")
    levels = E2M1.to(device)
    scales = e4m3fn_positive_scales(device)
    rotations = act.MethodRotations(
        workdir, MODEL, "nar", 0, args.seed, layers, dimensions, device
    )
    new_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    cv_sums = {(site, method): 0.0 for site in SITES for method in METHODS}
    cv_counts = {(site, method): 0 for site in SITES for method in METHODS}

    for site in SITES:
        n = int(meta["hidden_size"] if site == "q_input" else meta["intermediate_size"])
        target_site = "qkv" if site == "q_input" else "down"
        rows_per_layer = meta["site_shapes"][site][0] * math.ceil(
            meta["site_shapes"][site][1] / args.sample_stride
        )
        total_blocks = int(rows_per_layer * (n // BLOCK) * layers)
        accumulators = {
            rule: TailAccumulator(total_blocks, device) for rule in SCALE_RULES
        }
        for layer in range(layers):
            mmap = ext._open_site(wide, meta, site, layer)
            value = ext._sample_site_tokens(mmap, args.sample_stride, device)
            signs = _random_signs(n, layer, site, args.seed, device)
            factor = rotations.factors[(target_site, layer)]
            # The old rows are transformed only for the new energy-dispersion
            # diagnostic. Their FP4 quantization outputs are never recomputed.
            transformed = {
                "hadamard_b128": h128_rows(value, signs),
                "hadamard_b128_pi": pi_h128_rows(value, signs, factor),
                "nar_b128": factor.apply(value, signs),
                "hadamard_b16": transform_rows(
                    "hadamard_b16", value, signs, factor, factor
                ),
            }
            for method, current in transformed.items():
                cv_sum, cv_count = mean_group_energy_cv(current)
                cv_sums[(site, method)] += cv_sum
                cv_counts[(site, method)] += cv_count
                cv_rows.append({
                    "model": MODEL,
                    "site": site,
                    "layer": layer,
                    "method": method,
                    "mean_token_group128_energy_cv": cv_sum / cv_count,
                    "tokens": cv_count,
                })
            pi_blocks = transformed["hadamard_b128_pi"].reshape(-1, BLOCK)
            signal = pi_blocks.square().sum(-1)
            errors = {
                "absmax": block_errors_absmax(pi_blocks, levels),
                "mse_optimal": block_errors_mse_optimal(pi_blocks, levels, scales),
            }
            for rule, error in errors.items():
                accumulators[rule].add(error, signal)
            del mmap, value, transformed, pi_blocks, signal, errors
            gc.collect()
            torch.cuda.empty_cache()
        for rule, accumulator in accumulators.items():
            new_rows.append({
                "model": MODEL,
                "site": site,
                "method": "hadamard_b128_pi",
                "mixing_width": MIXING_WIDTH,
                "alignment": False,
                "frozen_nar_permutation": True,
                "fp4_block_size": BLOCK,
                "scale_rule": rule,
                **accumulator.summarize(),
            })
        del accumulators
        gc.collect()

    followup = base.read_csv(result_dir / "e15_followup_block_distribution.csv")
    h128 = base.read_csv(result_dir / "e15_h128_control.csv")
    combined: list[dict[str, Any]] = []
    for site in SITES:
        for rule in SCALE_RULES:
            for method in METHODS:
                if method == "hadamard_b128_pi":
                    source: dict[str, Any] = next(
                        row for row in new_rows
                        if row["site"] == site and row["scale_rule"] == rule
                    )
                    provenance = "new_pi_only_row"
                else:
                    source = _old_row(followup, h128, site, method, rule)
                    provenance = "reused_existing_quantization_row"
                combined.append({
                    "model": MODEL,
                    "site": site,
                    "scale_rule": rule,
                    "method": method,
                    "global_nmse": float(source["global_nmse"]),
                    "worst_1pct_error_share": float(source["worst_1pct_error_share"]),
                    "mean_token_group128_energy_cv": (
                        cv_sums[(site, method)] / cv_counts[(site, method)]
                    ),
                    "quantization_provenance": provenance,
                })

    def nmse(method: str) -> float:
        return next(
            float(row["global_nmse"]) for row in combined
            if row["site"] == "down_input"
            and row["scale_rule"] == "mse_optimal"
            and row["method"] == method
        )

    h128_value = nmse("hadamard_b128")
    pi_value = nmse("hadamard_b128_pi")
    aligned_value = nmse("nar_b128")
    gap = h128_value - aligned_value
    recovery = (h128_value - pi_value) / gap
    if recovery >= RECOVERY_HIGH:
        attribution = "load_balancing_across_groups"
    elif recovery <= RECOVERY_LOW:
        attribution = "directional_separation_by_GV"
    else:
        attribution = "mixed_contributions_no_single_cause_label"
    decision = {
        "site": "down_input",
        "scale_rule": "mse_optimal",
        "h128_only_nmse": h128_value,
        "h128_pi_nmse": pi_value,
        "aligned_h128_nmse": aligned_value,
        "total_gap": gap,
        "pi_improvement": h128_value - pi_value,
        "remaining_GV_improvement": pi_value - aligned_value,
        "pi_gap_recovery_fraction": recovery,
        "thresholds": {"directional_at_or_below": RECOVERY_LOW, "load_balance_at_or_above": RECOVERY_HIGH},
        "attribution": attribution,
    }
    base.write_csv(result_dir / "e15_pi_only.csv", new_rows)
    base.write_csv(result_dir / "e15_pi_group_energy_per_layer.csv", cv_rows)
    base.write_csv(result_dir / "e15_alignment_width_pi_control.csv", combined)
    base.atomic_json(done, {
        "model": MODEL,
        "new_row": "H128 + frozen NAR Pi only; no G(V)",
        "existing_rows_rerun": False,
        "existing_rows_recomputed_only_for_new_diagnostic": "per-token group-128 signal-energy CV",
        "pi_source": "source_order/target_order from frozen preregistered NAR-b128 factor checkpoint",
        "decision_rule_preregistered_before_run": (
            "down_input MSE-optimal Pi recovery >=0.70 => load balancing; <=0.30 => G(V) directional separation; otherwise mixed"
        ),
        "decision": decision,
        "group_energy_dispersion": "population coefficient of variation across per-128-group signal energies, then mean over identical sampled tokens and layers",
        "sample_stride": args.sample_stride,
        "seed": args.seed,
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
