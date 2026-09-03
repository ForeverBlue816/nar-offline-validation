#!/usr/bin/env python3
"""Write completed one-seed E16 robustness and diagnostic results to report.md."""

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

    diagnostics = {}
    offline = {}
    for key, _ in MODELS:
        result_dir = workdir / "results" / key
        dc_path = result_dir / "e16_dc_alignment_summary.csv"
        offline_path = result_dir / "e16_offline_per_layer.csv"
        if not dc_path.exists() or not offline_path.exists():
            continue
        with dc_path.open(newline="") as handle:
            dc_rows = list(csv.DictReader(handle))
        diagnostics[key] = {}
        for method in ("hadamard", "smoothquant_hadamard", "duquant_style", "nar"):
            diagnostics[key][method] = {
                row["direction"]: float(row["mean_s_i_across_layers"])
                for row in dc_rows
                if row["method"] == method
            }
        with offline_path.open(newline="") as handle:
            offline_rows = list(csv.DictReader(handle))
        offline[key] = {}
        for method in ("hadamard", "smoothquant_hadamard"):
            for site in ("qkv", "down"):
                selected = [
                    row for row in offline_rows
                    if row["method"] == method and row["site"] == site
                ]
                offline[key][method, site] = (
                    sum(float(row["mean_group_range"]) for row in selected) / len(selected),
                    sum(float(row["nmse"]) for row in selected) / len(selected),
                )

    report_path = workdir / "report.md"
    report = report_path.read_text()
    marker = "\n# E16 — post-hoc SmoothQuant robustness"
    next_marker = "\n# E17 — fused one-pass R4"
    prefix, suffix = report, ""
    if marker in report:
        prefix, old_tail = report.split(marker, 1)
        if next_marker in old_tail:
            _, suffix_tail = old_tail.split(next_marker, 1)
            suffix = next_marker + suffix_tail
        prefix = prefix.rstrip() + "\n"
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
    if diagnostics:
        lines.extend([
            "## E16 offline location diagnostic\n",
            "The paired offline check uses the same frozen 128 calibration sequences and compares plain Hadamard with SmoothQuant(alpha=0.5)+Hadamard. Values are arithmetic means over layers. SmoothQuant sharply contracts the transformed ranges, but slightly *increases* normalized INT4 error at both sites; range contraction alone therefore does not explain accuracy.\n",
            "| model | site | Had range | SQ+Had range | Had NMSE | SQ+Had NMSE |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for key, label in MODELS:
            for site, site_label in (("qkv", "q_input"), ("down", "down_input")):
                had_range, had_nmse = offline[key]["hadamard", site]
                sq_range, sq_nmse = offline[key]["smoothquant_hadamard", site]
                lines.append(
                    f"| {label} | {site_label} | {had_range:.6f} | {sq_range:.6f} | "
                    f"{had_nmse:.7f} | {sq_nmse:.7f} |"
                )
        lines.extend([
            "\n## E16 DC-alignment diagnostic\n",
            "Here `s_i = ||P_DC R v_i||^2 / ||v_i||^2`, averaged across layers, for the frozen top-eight q_input second-moment directions. The top-channel column uses the highest-magnitude calibration channel. SmoothQuant+Hadamard is non-orthogonal, so its denominator remains the original direction energy as pre-specified. Official DuQuant is omitted under the later citation-only/no-local-run amendment; `DuQuant-style` is the frozen E11 construction.\n",
        ])
        method_labels = {
            "hadamard": "Hadamard",
            "smoothquant_hadamard": "SmoothQuant+Hadamard",
            "duquant_style": "DuQuant-style",
            "nar": "NAR",
        }
        for key, label in MODELS:
            lines.extend([
                f"### {label}\n",
                "| method | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | top-8 mean | top channel |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for method, method_label in method_labels.items():
                values = diagnostics[key][method]
                top8 = [values[f"v{i}"] for i in range(1, 9)]
                fields = " | ".join(f"{value:.4f}" for value in top8)
                lines.append(
                    f"| {method_label} | {fields} | {sum(top8) / 8:.4f} | "
                    f"{values['top_magnitude_channel']:.4f} |"
                )
            lines.append("")
        lines.extend([
            "\nNAR places essentially all top-eight eigendirection energy into the group-128 DC subspace on both models (mean 1.0000). DuQuant-style captures only 0.4018 on 3B and 0.5249 on 8B because it greedily aligns one magnitude channel rather than the top eigenspace; accordingly, its selected top channel scores 1.0000. Plain Hadamard scores 0.0024/0.0054, while SmoothQuant+Hadamard scores just 0.00023/0.00027. This directly supports the claimed structural distinction: NAR explicitly uses the asymmetric quantizer's zero-point null space; the fair baselines do not reproduce its top-eigenspace alignment.\n",
            "Exact per-layer and aggregate rows are in `e16_offline_per_layer.csv`, `e16_dc_alignment_per_layer.csv`, and `e16_dc_alignment_summary.csv` under each model's result directory.\n",
        ])
    body = prefix.rstrip() + "\n\n" + "\n".join(lines)
    if suffix:
        body += "\n" + suffix
    report_path.write_text(body.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    build(args.workdir.resolve())


if __name__ == "__main__":
    main()
