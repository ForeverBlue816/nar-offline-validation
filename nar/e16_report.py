#!/usr/bin/env python3
"""Append completed one-seed E16 SmoothQuant robustness results to report.md."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODELS = (("llama32_3b", "Llama-3.2-3B"), ("llama31_8b", "Llama-3.1-8B"))


def build(workdir: Path) -> None:
    rows = []
    for key, label in MODELS:
        path = workdir / "results" / key / "e16_smoothquant_summary.csv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["model_label"] = label
                rows.append(row)
    report_path = workdir / "report.md"
    report = report_path.read_text()
    marker = "\n# E16 — post-hoc SmoothQuant robustness"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    lines = [
        "# E16 — post-hoc SmoothQuant robustness\n",
        "This section is explicitly post-hoc and uses one seed, following the amended single-seed execution rule. The E11 Hadamard and NAR rows are reused without rerunning; all variants use the same 64 WikiText-2 chunks, both activation sites unless noted, asymmetric group-128 INT4, and 4.25 effective activation bits/value. Alpha is not swept beyond the two requested robustness points.\n",
        "| model | variant | alpha | smoothing sites | PPL | delta vs Hadamard | delta vs NAR kmax |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    names = {
        "smoothquant_a065_both": "SmoothQuant+Hadamard",
        "smoothquant_a080_both": "SmoothQuant+Hadamard",
        "smoothquant_a050_qkv_only": "SmoothQuant(qkv-only)+Hadamard",
    }
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {names[row['method']]} | {float(row['alpha']):.2f} | "
            f"{row['smoothing_sites']} | {float(row['ppl']):.6f} | "
            f"{float(row['ppl_delta_vs_hadamard']):+.6f} | {float(row['ppl_delta_vs_nar_kmax']):+.6f} |"
        )
    lines.extend([
        "\nIncreasing alpha to 0.65 and 0.80 degrades both models monotonically relative to plain Hadamard. Restricting alpha=0.5 smoothing to its original q/k/v placement is marginally better than Hadamard (-0.0051 PPL on 3B, -0.0098 on 8B), showing that smoothing down_input caused most of E11's degradation; it still trails NAR kmax by +0.0469 and +0.0506 PPL. This robustness check therefore does not overturn E11. Confidence intervals are not estimable with one seed and are not implied.\n",
        "Exact rows are in `results/llama32_3b/e16_smoothquant_summary.csv` and `results/llama31_8b/e16_smoothquant_summary.csv`.\n",
    ])
    report_path.write_text(report.rstrip() + "\n\n" + "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    build(args.workdir.resolve())


if __name__ == "__main__":
    main()
