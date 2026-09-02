#!/usr/bin/env python3
"""Append completed E15 FP4 boundary result to report.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from . import experiment as base
except ImportError:
    import experiment as base


def _table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            f"{value:.6g}" if isinstance(value, (float, int)) else str(value) for value in row
        ) + " |")
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    root = workdir / "results" / "llama32_3b"
    done_path = root / "E15_DONE.json"
    if not done_path.exists():
        raise FileNotFoundError(done_path)
    done = json.loads(done_path.read_text())
    summary = pd.read_csv(root / "e15_fp4_summary.csv")
    comparison = pd.read_csv(root / "e15_fp4_comparison.csv")
    report_path = workdir / "report.md"
    report = report_path.read_text()
    marker = "\n# E15 — FP4 E2M1 framework boundary"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    verdict = "; ".join(
        f"{row.site}: NAR-Hadamard NMSE {row.mean_nar_minus_hadamard_nmse:+.6g}, "
        f"NAR better in {int(row.nar_better_layers)}/{int(row.layers)} layers"
        for row in comparison.itertuples(index=False)
    )
    section = [
        "# E15 — FP4 E2M1 framework boundary\n",
        "The retained E1c q_input/down_input tensors are reused without a model rerun. Every row is paired on the same every-128th-token sample. FP4 uses nearest finite E2M1 values with one max/6 E4M3FN scale per block of 16 and no zero-point. The Hadamard baseline is a fixed random-sign orthonormal H16 applied inside each aligned scale block. NMSE is the global squared-error/signal-energy ratio across layers; kurtosis is Pearson kurtosis of the transformed values.\n",
        _table(summary[["site", "method", "layers", "global_fp4_nmse", "mean_layer_fp4_nmse", "mean_transformed_pearson_kurtosis", "mean_relative_e4m3_scale_rounding_error"]]) + "\n",
        _table(comparison) + "\n",
        "The deliberately non-Gaussian invertible baseline is a fixed, seeded, randomly permuted diagonal transform with singular values exp(linspace(-ln4,ln4)), condition number 16. Its exact inverse is applied before NMSE; no parameter was tuned.\n",
        "Boundary verdict: " + verdict + ". The result confirms or refutes only whether NAR's zero-point-alignment benefit transfers to this no-zero-point FP4 framework; it is not used to revise the INT4 method result.\n",
    ]
    report_path.write_text(report.rstrip() + "\n\n" + "\n".join(section))
    base.atomic_json(workdir / "results" / "E15_REPORT_DONE.json", {
        "source": done, "comparison": comparison.to_dict(orient="records"), "no_tuning": True,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    return parser


if __name__ == "__main__":
    build(parser().parse_args())
