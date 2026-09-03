#!/usr/bin/env python3
"""E15 mixing-width control: H128 without NAR alignment, FP4 blocks of 16."""

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
    )
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
    )


MIXING_WIDTH = 128


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def h128_rows(value: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Seeded block-H128 only: no reflector, learned direction, or permutation."""
    shape = value.shape
    blocks = (value.float() * signs).reshape(-1, shape[-1] // MIXING_WIDTH, MIXING_WIDTH)
    return ext._fast_walsh_hadamard(blocks).reshape(shape)


def analyze(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E15 H128 control requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e15-h128-control")
    result_dir = workdir / "results" / MODEL
    done = result_dir / "E15_H128_CONTROL_DONE.json"
    if done.exists():
        return
    wide = workdir / "activations" / MODEL / "wide_cal_a"
    meta = _json(wide / "DONE.json")
    layers = int(meta["num_layers"])
    device = torch.device("cuda")
    levels = E2M1.to(device)
    scales = e4m3fn_positive_scales(device)
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for site in SITES:
        n = int(meta["hidden_size"] if site == "q_input" else meta["intermediate_size"])
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
            transformed = h128_rows(value, signs)
            blocks = transformed.reshape(-1, BLOCK)
            signal = blocks.square().sum(-1)
            errors = {
                "absmax": block_errors_absmax(blocks, levels),
                "mse_optimal": block_errors_mse_optimal(blocks, levels, scales),
            }
            for rule, error in errors.items():
                accumulators[rule].add(error, signal)
                layer_rows.append({
                    "model": MODEL,
                    "site": site,
                    "layer": layer,
                    "method": "hadamard_b128",
                    "mixing_width": MIXING_WIDTH,
                    "fp4_block_size": BLOCK,
                    "scale_rule": rule,
                    "layer_global_nmse": float(
                        error.double().sum() / signal.double().sum().clamp_min(1e-30)
                    ),
                    "blocks": error.numel(),
                })
            del mmap, value, transformed, blocks, signal, errors
            gc.collect()
            torch.cuda.empty_cache()
        for rule, accumulator in accumulators.items():
            rows.append({
                "model": MODEL,
                "site": site,
                "method": "hadamard_b128",
                "mixing_width": MIXING_WIDTH,
                "alignment": False,
                "fp4_block_size": BLOCK,
                "scale_rule": rule,
                **accumulator.summarize(),
            })
        del accumulators
        gc.collect()

    base.write_csv(result_dir / "e15_h128_control.csv", rows)
    base.write_csv(result_dir / "e15_h128_control_per_layer.csv", layer_rows)
    base.atomic_json(done, {
        "model": MODEL,
        "method": "seeded block-H128 only",
        "alignment": False,
        "permutation": False,
        "mixing_width": MIXING_WIDTH,
        "fp4_block_size": BLOCK,
        "scale_rules": {
            "absmax": "same E15 rounded E4M3FN max(abs(x))/6 rule",
            "mse_optimal": "same exact exhaustive search over all 126 positive finite E4M3FN scales",
        },
        "paired": "same frozen E1c vectors, sample stride, and signs as E15 follow-up",
        "sample_stride": args.sample_stride,
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
