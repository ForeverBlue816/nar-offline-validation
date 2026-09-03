#!/usr/bin/env python3
"""Append completed Qwen3-8B E18 generality results to report.md."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build(workdir: Path) -> None:
    root = workdir / "results" / "qwen3_8b"
    rows = list(csv.DictReader((root / "e18_summary.csv").open(newline="")))
    done = json.loads((root / "E18_DONE.json").read_text())
    values = {row["method"]: float(row["ppl"]) for row in rows}
    gap = values["hadamard"] - values["bf16"]
    recovery8 = (values["hadamard"] - values["nar_k8"]) / gap
    recovery_max = (values["hadamard"] - values["nar_kmax"]) / gap
    report_path = workdir / "report.md"
    report = report_path.read_text()
    marker = "\n# E18 — Qwen3-8B generality"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    labels = {"bf16": "bf16", "hadamard": "Hadamard", "nar_k8": "NAR k=8", "nar_kmax": "NAR k=max"}
    lines = [
        "# E18 — Qwen3-8B generality\n",
        f"Qwen3-8B has head_dim={done['head_dim']}, hidden={done['hidden_size']}, and intermediate={done['intermediate_size']}; group 128 therefore provides {done['slot_counts']['qkv']} q/k/v-input slots and {done['slot_counts']['down']} down-input slots. The paired both-sites E5 protocol is unchanged: 128 calibration sequences, 64 WikiText-2 test chunks at context 2048, bf16 weights/KV/all other activations, and dynamic asymmetric group-128 INT4 (4.25 effective bits/value) only at the two target activation sites. One seed is used under the amended execution rule.\n",
        "| method | PPL | delta vs bf16 | delta vs Hadamard | effective bits/value |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {labels[row['method']]} | {float(row['ppl']):.6f} | "
            f"{float(row['ppl_delta_vs_bf16']):+.6f} | "
            f"{float(row['ppl_delta_vs_hadamard']):+.6f} | {float(row['effective_bits_per_value']):.2f} |"
        )
    lines.extend([
        f"\nNAR k=8 recovers only {100 * recovery8:.1f}% of the Hadamard-to-bf16 gap on this model, substantially less than on the Llama models. NAR k=max recovers {100 * recovery_max:.1f}% and yields a PPL 0.3040 below the bf16 row. The latter is reported as a surprising one-seed observation, not a claim that quantization improves the base model: no seed-level CI is estimable, no hyperparameter was tuned, and no confirmation rerun was performed. The paired chunks and weight-fold audits are retained for diagnosis.\n",
        "Exact outputs are in `results/qwen3_8b/e18_per_sequence.csv`, `e18_summary.csv`, and `E18_DONE.json`.\n",
    ])
    report_path.write_text(report.rstrip() + "\n\n" + "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    build(args.workdir.resolve())


if __name__ == "__main__":
    main()
