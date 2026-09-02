#!/usr/bin/env python3
"""Append the completed E12 compact-WY result to report.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from . import experiment as base
except ImportError:
    import experiment as base


def _format(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, int)):
        return f"{value:.6g}"
    return str(value)


def _table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    root = workdir / "results" / "llama32_3b"
    done_path = root / "E12_DONE.json"
    if not done_path.exists():
        raise FileNotFoundError(done_path)
    done = json.loads(done_path.read_text())
    verify = pd.read_csv(root / "e12_wy_verification.csv")[[
        "k", "wy_vs_sequential_g_max_abs_error", "wy_vs_sequential_g_relative_l2_error",
        "wy_full_vs_dense_max_abs_error", "wy_full_vs_dense_relative_l2_error",
    ]]
    timing = pd.read_csv(root / "e12_wy_online_cost.csv")[[
        "tokens", "k", "r4_flop_ratio_vs_matmul", "wy_g_ms", "r4_wy_ms",
        "unfused_hadamard_ms", "down_matmul_ms", "r4_wall_ratio_vs_unfused_hadamard",
        "r4_wall_ratio_vs_down_matmul", "r4_under_10pct_matmul_wall",
    ]]
    report = (workdir / "report.md").read_text()
    marker = "\n# E12 — compact-WY deployable R4"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    at_2048 = timing[timing.tokens == 2048]
    pass_2048 = bool(at_2048.r4_under_10pct_matmul_wall.all())
    at_one = timing[timing.tokens == 1]
    launch_fail = not bool(at_one.r4_under_10pct_matmul_wall.all())
    sections = [
        "# E12 — compact-WY deployable R4\n",
        "The sequential Householder product is represented exactly in compact WY form as G=I-WY^T, with W,Y in R^(8192 x k). Applying G therefore uses two small matrix multiplications; the complete R4 then applies the fixed permutation/sign and block H128. Results cover the E11 knee ranks k=16/32 and the original k=64. The unfused Hadamard reference is the same staged PyTorch FWHT as E6, not a custom fused kernel. Rotation timing is fp32 and down_proj timing is bf16, matching E6.\n",
        _table(verify) + "\n",
        _table(timing) + "\n",
        ("**2048-token engineering gate: PASS.** Every measured compact-WY rank is under 10% of down_proj wall-clock.\n" if pass_2048 else "**2048-token engineering gate: FAIL.** At least one measured compact-WY rank exceeds 10% of down_proj wall-clock.\n"),
    ]
    if launch_fail:
        sections.append("At one token the WY form is not under 10%; this regime is launch-bound. The unfused Hadamard has the same qualitative problem, so the single-token result is reported rather than hidden.\n")
    sections.append("The FLOP ratio includes both WY matmuls plus the block-Hadamard/sign term; wall-clock ratios include the entire R4. No fused kernel or timing-specific tuning was used.\n")
    (workdir / "report.md").write_text(report.rstrip() + "\n\n" + "\n".join(sections))
    base.atomic_json(workdir / "results" / "E12_REPORT_DONE.json", {
        "under_10pct_at_2048_all_ranks": pass_2048,
        "one_token_launch_bound_failure": launch_fail,
        "source": done,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    return parser


if __name__ == "__main__":
    build(parser().parse_args())
