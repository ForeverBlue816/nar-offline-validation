#!/usr/bin/env python3
"""Build the E16 null-space-capture main and appendix figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HAD = "#E69F00"
DUQ = "#009E73"
NAR = "#0072B2"
GROUP = 128
SEED = 20260902
MODELS = {
    "llama31_8b": {"label": "Llama-3.1-8B", "layers": 32, "qkv": 4096, "down": 14336},
    "llama32_3b": {"label": "Llama-3.2-3B", "layers": 28, "qkv": 3072, "down": 8192},
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def import_project(repo: Path) -> tuple[object, object, object]:
    sys.path.insert(0, str(repo))
    from nar import activation_experiments as act
    from nar import e11_fair_baselines as e11
    from nar import e16_diagnostics as e16

    return act, e11, e16


def import_alignment_helper() -> object:
    helper = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(helper))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def derive_capture(repo: Path, workdir: Path, csv_path: Path) -> dict[str, float]:
    act, e11, e16 = import_project(repo)
    output: list[dict[str, object]] = []
    crosscheck: dict[str, float] = {}
    device = torch.device("cpu")
    for model, spec in MODELS.items():
        stats_path = workdir / "activations" / model / "e11_calibration" / "channel_stats.pt"
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        dims = {"qkv": spec["qkv"], "down": spec["down"]}
        had = e11.Transform("hadamard_g128_asym", model, workdir, 0, SEED, spec["layers"], dims, device, stats)
        duq = e11.Transform("duquant_style_g128_asym", model, workdir, 0, SEED, spec["layers"], dims, device, stats)
        existing = pd.read_csv(repo / "results" / model / "e16_dc_alignment_per_layer.csv")
        existing = existing[existing.direction.str.fullmatch(r"v[1-8]") & existing.method.isin(["hadamard", "duquant_style", "nar"])]
        qkv_recomputed: list[dict[str, object]] = []
        for site in ("qkv", "down"):
            for layer in range(spec["layers"]):
                factor_path = workdir / "activations" / model / "activation_factors" / f"{site}_layer_{layer:02d}.pt"
                factor = act.RotationFactor.load(factor_path, device)
                directions = e16.reconstruct_directions(factor, 8)
                for method, transform in (("hadamard", had), ("duquant_style", duq)):
                    transformed = transform.activation(site, layer, directions)
                    scores = e16.dc_fraction(transformed, directions, GROUP)
                    for index, score in enumerate(scores, 1):
                        row = {
                            "model": model,
                            "site": site,
                            "method": method,
                            "layer": layer,
                            "direction_index": index,
                            "s_i": float(score),
                            "source": "derived from frozen E16 factors and E11 transform statistics",
                        }
                        output.append(row)
                        if site == "qkv":
                            qkv_recomputed.append(row)
                transformed = factor.apply(directions, had.signs[(site, layer)])
                scores = e16.dc_fraction(transformed, directions, GROUP)
                for index, score in enumerate(scores, 1):
                    row = {
                        "model": model,
                        "site": site,
                        "method": "nar",
                        "layer": layer,
                        "direction_index": index,
                        "s_i": float(score),
                        "source": "derived from frozen E16 factors and E11 transform statistics",
                    }
                    output.append(row)
                    if site == "qkv":
                        qkv_recomputed.append(row)
        recomputed = pd.DataFrame(qkv_recomputed).sort_values(["method", "layer", "direction_index"])
        expected = existing.assign(direction_index=existing.direction.str[1:].astype(int)).sort_values(["method", "layer", "direction_index"])
        if len(recomputed) != len(expected):
            raise AssertionError(f"{model}: E16 qkv cross-check row count mismatch")
        max_error = float(np.max(np.abs(recomputed.s_i.to_numpy() - expected.s_i.to_numpy())))
        crosscheck[model] = max_error
        if max_error > 2e-5:
            raise AssertionError(f"{model}: E16 qkv cross-check max error {max_error}")
    pd.DataFrame(output).to_csv(csv_path, index=False)
    return crosscheck


def panel_letter(ax: plt.Axes, letter: str) -> None:
    offset = mpl.transforms.ScaledTranslation(-11 / 72, 5 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + offset, fontsize=8.5, fontweight="bold", va="bottom")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_model(data: pd.DataFrame, model: str, outbase: Path) -> dict[str, float]:
    configure_style()
    spec = MODELS[model]
    subset = data[data.model.eq(model)]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.91, wspace=0.16)
    methods = [
        ("hadamard", "Hadamard", HAD, 1.15),
        ("duquant_style", "DuQuant-style", DUQ, 1.3),
        ("nar", "NAR", NAR, 1.9),
    ]
    summary: dict[str, float] = {}
    for panel, (ax, site, site_name) in enumerate(zip(axes, ("qkv", "down"), ("q/k/v input", "down-projection input"))):
        site_data = subset[subset.site.eq(site)]
        directions = sorted(site_data.direction_index.unique())
        x = np.asarray(directions)
        ax.axhline(1.0, color="#8D8D8D", lw=0.65, ls=(0, (3, 2)), zorder=0)
        random_ref = (int(spec[site]) // GROUP) / int(spec[site])
        ax.axhline(random_ref, color="#8D8D8D", lw=0.65, ls=(0, (3, 2)), zorder=0)
        for method, name, color, linewidth in methods:
            method_data = site_data[site_data.method.eq(method)]
            pivot = method_data.pivot(index="layer", columns="direction_index", values="s_i").reindex(columns=directions)
            mean = pivot.mean(axis=0).to_numpy()
            low = pivot.quantile(0.10, axis=0).to_numpy()
            high = pivot.quantile(0.90, axis=0).to_numpy()
            method_mean = float(method_data.s_i.mean())
            summary[f"{site}_{method}_mean"] = method_mean
            ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
            ax.plot(x, mean, color=color, lw=linewidth, marker="o", ms=2.2 if method != "nar" else 2.8, mec="white", mew=0.3)
            y_text = method_mean
            if method == "nar":
                y_text = 0.965
            elif method == "hadamard":
                y_text = 0.040
            elif site == "down":
                y_text = 0.125
            display_name = "DuQuant" if method == "duquant_style" else name
            ax.text(12.65, y_text, f"{display_name}  {method_mean:.3f}", color=color, fontsize=7.0, ha="right", va="center", fontweight="bold" if method == "nar" else "normal")
        ax.text(0.03, 0.87, site_name, transform=ax.transAxes, fontsize=7.7, fontweight="bold", va="top")
        ax.text(0.03, 0.77, "line: layer mean  ·  band: 10–90%", transform=ax.transAxes, fontsize=7.0, color="#555555", va="top")
        ax.text(0.40, 0.975, "fully absorbed", transform=ax.transAxes, fontsize=7.0, color="#666666", ha="center", va="bottom")
        ax.text(12.65, 0.26, f"random orthogonal\nG/d = {random_ref:.4f}", color="#666666", fontsize=7.0, ha="right", va="center")
        ax.set_xlim(1, 13.0)
        ax.set_ylim(-0.015, 1.065)
        ax.set_xticks(x)
        ax.set_xlabel("second-moment direction index")
        if panel == 0:
            ax.set_ylabel("fraction captured by DC null space, sᵢ")
        clean_axis(ax)
        panel_letter(ax, chr(ord("a") + panel))
    fig.text(0.995, 0.006, f"{spec['label']} · frozen calibration · {spec['layers']} layers", fontsize=7.0, color="#555555", ha="right", va="bottom")
    fig.canvas.draw()
    require_matplotlib_panel_alignment = import_alignment_helper()
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes),
        panel_ids=["a", "b"],
        row_groups=[{"id": "sites", "panels": ["a", "b"]}],
        require_panel_labels=True,
        json_out=outbase.with_suffix(".alignment.json"),
        overlay_svg=outbase.with_suffix(".alignment-overlay.svg"),
    )
    fig.savefig(outbase.with_suffix(".svg"))
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    csv_path = here / "fig2_capture.csv"
    crosscheck = derive_capture(args.repo.resolve(), args.workdir.resolve(), csv_path)
    data = pd.read_csv(csv_path)
    main_summary = render_model(data, "llama31_8b", here / "fig2_null_space_capture")
    appendix_summary = render_model(data, "llama32_3b", here / "fig2_null_space_capture_3b")
    metadata = {"qkv_crosscheck_max_abs": crosscheck, "main_summary": main_summary, "appendix_summary": appendix_summary}
    (here / "fig2_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
