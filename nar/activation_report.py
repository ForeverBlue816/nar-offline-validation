#!/usr/bin/env python3
"""Build the final E5-E8 report and figures from completed frozen artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import experiment as base
except ImportError:
    import experiment as base


MODELS = ("llama32_3b", "llama32_1b", "llama31_8b")
MODEL_LABELS = {"llama32_3b": "Llama-3.2-3B", "llama32_1b": "Llama-3.2-1B", "llama31_8b": "Llama-3.1-8B"}
SITE_LABELS = {"qkv_only": "q/k/v only", "down_only": "down_proj only", "both": "both sites", "none": "none"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _table(frame: pd.DataFrame) -> str:
    return base.md_table(list(frame.columns), frame.itertuples(index=False, name=None))


def make_plots(workdir: Path, e5: pd.DataFrame, e7_rank: pd.DataFrame, e8: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_dir = workdir / "results" / "activation"
    result_dir.mkdir(parents=True, exist_ok=True)
    methods = ("identity", "hadamard", "nar")
    colors = {"identity": "#777777", "hadamard": "#2878b5", "nar": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    for axis, model in zip(axes, MODELS):
        subset = e5[(e5.model == model) & (e5.site != "none")]
        x = np.arange(3)
        width = 0.24
        for offset, method in enumerate(methods):
            rows = subset[subset.method == method].set_index("site")
            values = [float(rows.loc[site, "ppl_delta_vs_bf16"]) for site in ("qkv_only", "down_only", "both")]
            axis.bar(x + (offset - 1) * width, values, width, label=method, color=colors[method])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, ["qkv", "down", "both"])
        axis.set_title(MODEL_LABELS[model])
        axis.set_ylabel("PPL delta vs bf16")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].legend()
    fig.suptitle("E5 activation-only INT4 perplexity proxy")
    fig.tight_layout()
    fig.savefig(result_dir / "e5_ppl_delta.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, b in zip(axes, (32, 64, 128)):
        subset = e7_rank[e7_rank.b == b]
        for layer, rows in subset.groupby("layer"):
            axis.plot(rows.k, rows.mean_group_range, color="0.78", linewidth=0.7, alpha=0.55)
        means = subset.groupby("k", as_index=False).mean(numeric_only=True)
        axis.plot(means.k, means.mean_group_range, color="#d62728", marker="o", linewidth=2)
        axis.set_title(f"V cache, b={b}")
        axis.set_xlabel("absorbed rank k")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean group range")
    fig.tight_layout()
    fig.savefig(result_dir / "e7_v_range_vs_k.png", dpi=180)
    plt.close(fig)

    held = e8[(e8.split == "heldout_cal_b") & (e8.method == "range_direct")].sort_values("layer")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(held.layer, held.range_delta_vs_second_moment, color="#d62728")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("held-out range delta")
    axes[1].bar(held.layer, held.nmse_delta_vs_second_moment, color="#2878b5")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("held-out NMSE delta")
    for axis in axes:
        axis.set_xlabel("layer")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("E8 range-direct minus second-moment NAR (negative is better)")
    fig.tight_layout()
    fig.savefig(result_dir / "e8_heldout_deltas.png", dpi=180)
    plt.close(fig)


def build(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    for model in MODELS:
        required = [workdir / "results" / model / "E5_DONE.json", workdir / "results" / model / "e5_summary.csv"]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
    for name in ("E6_DONE.json", "E7_DONE.json", "E8_DONE.json"):
        path = workdir / "results" / "llama32_3b" / name
        if not path.exists():
            raise FileNotFoundError(path)
    e5 = pd.concat([pd.read_csv(workdir / "results" / model / "e5_summary.csv") for model in MODELS], ignore_index=True)
    e6 = pd.read_csv(workdir / "results" / "llama32_3b" / "e6_online_cost.csv")
    e7 = pd.read_csv(workdir / "results" / "llama32_3b" / "e7_summary.csv")
    e7_rank = pd.read_csv(workdir / "results" / "llama32_3b" / "e7_range_vs_k.csv")
    e7_fit = pd.read_csv(workdir / "results" / "llama32_3b" / "e7_energy_fit.csv")
    e7_inv = pd.read_csv(workdir / "results" / "llama32_3b" / "e7_o_proj_fold_invariance.csv")
    e8 = pd.read_csv(workdir / "results" / "llama32_3b" / "e8_per_layer.csv")
    e8_summary = pd.read_csv(workdir / "results" / "llama32_3b" / "e8_summary.csv")
    e8_done = _json(workdir / "results" / "llama32_3b" / "E8_DONE.json")
    make_plots(workdir, e5, e7_rank, e8)

    old_report = (workdir / "report.md").read_text()
    marker = "\n# Activation continuation — E5–E8"
    if marker in old_report:
        old_report = old_report.split(marker, 1)[0].rstrip() + "\n"

    e5_display = e5.copy()
    e5_display["model"] = e5_display.model.map(MODEL_LABELS)
    e5_display["site"] = e5_display.site.map(SITE_LABELS)
    e5_display = e5_display[[
        "model", "site", "method", "mean_ppl", "ppl_delta_vs_bf16",
        "paired_ppl_delta_vs_hadamard", "paired_90ci_low_vs_hadamard", "paired_90ci_high_vs_hadamard",
        "paired_ppl_delta_vs_identity", "paired_90ci_low_vs_identity", "paired_90ci_high_vs_identity",
    ]]
    e6_display = e6[[
        "tokens", "householder_reflections", "nar_flops_per_token", "down_matmul_flops_per_token",
        "nar_flop_ratio_vs_matmul", "nar_ms", "hadamard_ms", "down_matmul_ms",
        "nar_wall_ratio_vs_hadamard", "nar_wall_ratio_vs_down_matmul", "nar_exceeds_10pct_matmul_wall",
    ]]
    e7_display = e7[["b", "method", "layers", "mean_group_range", "mean_relative_quantization_error_nmse",
                     "mean_range_reduction_vs_hadamard", "mean_nmse_delta_vs_hadamard"]]
    e8_display = e8_summary[["split", "method", "layers", "mean_group_range",
                             "mean_relative_quantization_error_nmse", "mean_range_delta_vs_second_moment",
                             "mean_nmse_delta_vs_second_moment"]]

    nar_rows = e5[(e5.method == "nar") & (e5.site != "none")]
    e5_sentences = []
    for row in nar_rows.itertuples(index=False):
        e5_sentences.append(
            f"{MODEL_LABELS[row.model]} {SITE_LABELS[row.site]}: NAR-Hadamard {row.paired_ppl_delta_vs_hadamard:.6f} "
            f"(90% CI [{row.paired_90ci_low_vs_hadamard:.6f}, {row.paired_90ci_high_vs_hadamard:.6f}]), "
            f"NAR-identity {row.paired_ppl_delta_vs_identity:.6f}"
        )
    e6_fail = bool(e6.nar_exceeds_10pct_matmul_wall.any())
    e8_decision = e8_done["decision"]
    e7_inv_max = float(e7_inv.max_abs_error.max())
    fold_max = max(
        float(pd.read_csv(workdir / "results" / model / "e5_weight_fold_audit.csv").max_relative_error.max())
        for model in MODELS
    )

    sections: list[str] = []
    sections.append("# Activation continuation — E5–E8\n")
    sections.append("## Scope decision\n")
    sections.append("K is closed: KIVI-style per-channel quantization remains the clear K result, consistent with the zero-point null-space interpretation because channel-persistent outliers are constant along the token grouping axis. No additional K experiment was run. E5–E8 concern activation sites where per-token quantization is forced, plus the standard per-token V cache.\n")
    sections.append("## E5 — activation-only perplexity proxy\n")
    sections.append("Only post-RMSNorm q/k/v inputs and/or down_proj inputs are dynamically asymmetric group-128 INT4 fake-quantized. Scale and real-valued offset are fp16. Weights remain bf16; each activation rotation is folded algebraically into the corresponding q/k/v or down_proj weight rows. KV and all other activations remain bf16. Results use the same 64 WikiText-2 test chunks, three paired rotation seeds, and two-sided paired 90% Student-t intervals over seed-level PPL differences. The 8B model was included because a GPU was available.\n")
    sections.append(_table(e5_display) + "\n")
    sections.append("![E5 PPL deltas](results/activation/e5_ppl_delta.png)\n")
    sections.append("Paired NAR results:\n\n- " + "\n- ".join(e5_sentences) + "\n")
    sections.append(f"The maximum measured relative output discrepancy from storing the algebraically folded weights in bf16 was {fold_max:.6g}; this rounding is included in every rotated-method PPL result. No post-result tuning was performed.\n")

    sections.append("## E6 — factorized R4 online cost\n")
    sections.append("For the 3B down_proj input, G(V) is implemented as 64 sequential Householder reflections (below the 2k=128 cap), followed by a fixed permutation, signs, and block H128. The dense 8192x8192 matrix is materialized only for the equivalence audit, not the benchmark implementation.\n")
    sections.append(_table(e6_display) + "\n")
    dense = _json(workdir / "results" / "llama32_3b" / "E6_DONE.json")["dense_verification"]
    sections.append(f"Factorized-versus-dense fp32 verification: max absolute error {dense['max_abs_error']:.6g}, relative L2 error {dense['relative_l2_error']:.6g}. " + ("**Engineering check: FAIL. The measured unfused wall-clock cost exceeds 10% of down_proj matmul cost.**\n" if e6_fail else "**Engineering check: PASS at all measured token counts.**\n"))

    sections.append("## E7 — per-token V cache under NAR\n")
    sections.append("V is rotated within each KV head before dynamic asymmetric per-token INT4. The same R^T is folded blockwise into o_proj input columns (R2 position); the reported range/NMSE is offline and paired on identical V samples.\n")
    sections.append(_table(e7_display) + "\n")
    sections.append("The pooled fits `range(k)/range(0) = intercept + slope*sqrt(1-f)` are:\n")
    sections.append(_table(e7_fit) + "\n")
    sections.append(f"The o_proj fold identity has maximum fp64 absolute discrepancy {e7_inv_max:.6g}.\n")
    sections.append("![E7 V range versus k](results/activation/e7_v_range_vs_k.png)\n")

    sections.append("## E8 — one-shot range-direct refinement\n")
    sections.append("Starting from the frozen second-moment down_input V, each layer receives exactly 200 projected Riemannian-gradient steps with QR retraction, p=8, one seed, learning rate 0.05, and unit-Frobenius tangent normalization. Pi and signs remain fixed. Calibration uses cal-A samples; evaluation uses the next 128 disjoint WikiText-2 train chunks (cal-B). These choices were not changed after observing results.\n")
    sections.append(_table(e8_display) + "\n")
    sections.append(f"Held-out sign: mean range {'improved' if e8_decision['heldout_mean_range_improved'] else 'did not improve'} ({e8_decision['layers_range_improved']}/{e8_decision['layers']} layers); mean NMSE {'improved' if e8_decision['heldout_mean_nmse_improved'] else 'did not improve'} ({e8_decision['layers_nmse_improved']}/{e8_decision['layers']} layers). This diagnostic is reported as-is.\n")
    sections.append("![E8 held-out deltas](results/activation/e8_heldout_deltas.png)\n")

    sections.append("## Artifact retention and protocol integrity\n")
    sections.append("The original E1c q_input/down_input bf16 dumps remain untouched for E3/E4. E7 and E8 add separate V-cal-A and down-cal-B sampled dumps under project storage; none of the raw dumps enter Git. All result tables, completion metadata, factorization audits, plots, Slurm transcripts, and commands are published. E1/E2/E1c results were not rerun or modified.\n")
    (workdir / "report.md").write_text(old_report.rstrip() + "\n\n" + "\n".join(sections))
    decision = {
        "k_experiments_closed": True,
        "e5_models": list(MODELS),
        "e5_nar_rows": nar_rows.to_dict(orient="records"),
        "e6_online_cost_exceeds_10pct": e6_fail,
        "e6_dense_verification": dense,
        "e8_decision": e8_decision,
        "no_tuning": True,
    }
    base.atomic_json(workdir / "results" / "decision_activation.json", decision)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    return parser


if __name__ == "__main__":
    build(make_parser().parse_args())
