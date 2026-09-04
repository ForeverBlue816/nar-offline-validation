#!/usr/bin/env python3
"""Render Figure 3 as three final-size standalone matplotlib panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from figure_style import PALETTE, clean_2d_axis, configure_style, resolved_serif_family, save_panel


def new_panel(left: float = 0.28) -> tuple[plt.Figure, plt.Axes]:
    configure_style()
    fig, ax = plt.subplots(figsize=(1.85, 1.72))
    fig.subplots_adjust(left=left, right=0.96, bottom=0.25, top=0.95)
    clean_2d_axis(ax)
    ax.tick_params(labelsize=7.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    return fig, ax


def render_a(projections: pd.DataFrame, geometry: dict, outbase: Path) -> None:
    fig, ax = new_panel(left=0.30)
    p1 = projections.projection_v1.to_numpy()
    p2 = projections.projection_v2.to_numpy()
    p1 = (p1 - p1.mean()) / p1.std(ddof=0)
    p2 = (p2 - p2.mean()) / p2.std(ddof=0)
    ax.scatter(p1, p2, s=4.0, color=PALETTE["identity"], alpha=0.25, linewidths=0, rasterized=True)
    scale = 1.8
    had = geometry["arrows"]["hadamard"]
    had_x = scale * float(had["projection_v1"])
    had_y = scale * float(had["projection_v2"])
    ax.annotate(
        "", xy=(had_x, had_y), xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["hadamard"], "lw": 1.0, "mutation_scale": 5.5},
    )
    ax.text(-2.85, -1.25, f"Hadamard\n{had['in_plane_length']:.3f}", color=PALETTE["hadamard"],
            fontsize=7.5, ha="left", va="top", linespacing=0.9)
    prism = geometry["arrows"]["nar"]
    prism_x = scale * float(prism["projection_v1"])
    prism_y = scale * float(prism["projection_v2"])
    if prism_x < 0:
        prism_x, prism_y = -prism_x, -prism_y
    ax.annotate(
        "", xy=(prism_x, prism_y), xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["prismquant"], "lw": 1.8, "mutation_scale": 7.0},
    )
    ax.text(0.18, 0.78, f"PrismQuant\n{prism['in_plane_length']:.3f}", color=PALETTE["prismquant"],
            fontsize=7.5, ha="left", va="bottom", linespacing=0.9)
    limit = max(3.0, float(np.quantile(np.abs(np.concatenate([p1, p2])), 0.995)))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color=PALETTE["pane_edge"], lw=0.45, zorder=0)
    ax.axvline(0, color=PALETTE["pane_edge"], lw=0.45, zorder=0)
    ax.set_xlabel("v₁ projection (s.d.)", fontsize=8.0)
    ax.set_ylabel("v₂ projection (s.d.)", fontsize=8.0)
    save_panel(fig, outbase)


def render_b(eigenspace: pd.DataFrame, outbase: Path) -> None:
    fig, ax = new_panel(left=0.29)
    specs = (
        (1, PALETTE["identity"], "-", 1.8, "layer 1"),
        (13, PALETTE["identity"], (0, (3, 2)), 0.9, "layer 13"),
        (27, PALETTE["identity"], (0, (1, 2)), 1.0, "layer 27"),
    )
    endpoints: dict[int, float] = {}
    for layer, color, linestyle, width, _label in specs:
        part = eigenspace[eigenspace.layer.eq(layer)].sort_values("rank")
        if len(part) != 256:
            raise AssertionError(f"layer {layer}: rank-256 eigenspectrum missing")
        ax.plot(part["rank"], part.cumulative_fraction_total_energy, color=color, ls=linestyle, lw=width)
        endpoints[layer] = float(part.cumulative_fraction_total_energy.iloc[-1])
    ax.set_xlim(1, 256)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1, 256])
    ax.set_xticklabels(["1", "256"])
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    ax.set_xlabel("directions retained, k", fontsize=8.0)
    ax.set_ylabel("cumulative energy", fontsize=8.0)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.text(250, endpoints[27] + 0.045, "layer 27", color=PALETTE["text"], fontsize=7.5, ha="right", va="bottom")
    ax.text(250, endpoints[13] + 0.035, "layer 13", color=PALETTE["text"], fontsize=7.5, ha="right", va="bottom")
    ax.text(250, endpoints[1] - 0.035, "layer 1", color=PALETTE["text"], fontsize=7.5, ha="right", va="top")
    save_panel(fig, outbase)


def fit_law(law: pd.DataFrame) -> tuple[float, float, float]:
    x = law.sqrt_one_minus_f.to_numpy()
    y = law.range_ratio_vs_hadamard.to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = intercept + slope * x
    r_squared = 1.0 - float(np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2))
    return float(intercept), float(slope), r_squared


def scatter_sources(ax: plt.Axes, law: pd.DataFrame, inset: bool = False) -> None:
    specs = {
        "E1c activations": (PALETTE["prismquant"], "o", 6.0, "E1c activations"),
        "E7 V cache": (PALETTE["duquant"], "s", 8.0, "E7 V cache"),
        "E20 multi-slot": (PALETTE["prismquant_light"], "^", 9.0, "E20 multi-slot"),
    }
    for family, (color, marker, size, label) in specs.items():
        part = law[law.source_family.eq(family)]
        ax.scatter(
            part.sqrt_one_minus_f, part.range_ratio_vs_hadamard,
            color=color, marker=marker, s=size, alpha=0.35,
            edgecolors="none", rasterized=not inset, label=None if inset else label,
        )


def render_c(law: pd.DataFrame, outbase: Path) -> dict[str, float | bool]:
    fig, ax = new_panel(left=0.30)
    scatter_sources(ax, law)
    intercept, slope, r_squared = fit_law(law)
    grid = np.linspace(0, 1.02, 150)
    ax.plot([0, 1], [0, 1], color=PALETTE["reference"], lw=0.7, ls=(0, (3, 2)), label="identity")
    ax.plot(grid, intercept + slope * grid, color=PALETTE["reference"], lw=0.9)
    ax.text(0.035, 0.965, f"R² = {r_squared:.2f}", transform=ax.transAxes,
            color=PALETTE["text"], fontsize=7.5, ha="left", va="top")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(1.02, float(law.range_ratio_vs_hadamard.quantile(0.995)) * 1.04))
    ax.set_xlabel("√(1 − f)", fontsize=8.0)
    ax.set_ylabel("range / Hadamard", fontsize=8.0)
    crowded = int(((law.sqrt_one_minus_f > 0.90) & (law.range_ratio_vs_hadamard > 0.90)).sum()) >= 100
    if crowded:
        inset = ax.inset_axes([0.56, 0.56, 0.39, 0.34])
        scatter_sources(inset, law, inset=True)
        inset.plot([0.90, 1.0], [0.90, 1.0], color=PALETTE["reference"], lw=0.55, ls=(0, (3, 2)))
        inset.plot(grid, intercept + slope * grid, color=PALETTE["reference"], lw=0.7)
        inset.set_xlim(0.90, 1.005)
        inset.set_ylim(0.88, 1.02)
        inset.set_xticks([0.90, 1.00])
        inset.set_yticks([0.90, 1.00])
        inset.tick_params(labelsize=7.0, length=1.5, pad=1)
        for spine in inset.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color(PALETTE["pane_edge"])
    legend = ax.legend(
        loc="lower right", fontsize=7.0, handlelength=1.0, handletextpad=0.3,
        labelspacing=0.15, borderaxespad=0.15, frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    save_panel(fig, outbase)
    return {"fit_intercept": intercept, "fit_slope": slope, "fit_r_squared": r_squared, "corner_inset": crowded}


def make_preview(here: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(5.55, 1.72))
    for ax, letter in zip(axes, "abc"):
        ax.imshow(plt.imread(here / f"fig3{letter}.png"))
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01)
    fig.savefig(here / "fig3_preview.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    here = args.figures_dir.resolve()
    projections = pd.read_csv(here / "fig3_token_projections.csv")
    eigenspace = pd.read_csv(here / "fig3_eigenspace_r256.csv")
    law = pd.read_csv(here / "fig3_range_law.csv")
    geometry = json.loads((here / "fig3_geometry_metadata.json").read_text())
    render_a(projections, geometry, here / "fig3a")
    render_b(eigenspace, here / "fig3b")
    fit = render_c(law, here / "fig3c")
    make_preview(here)
    metadata = {
        "model": "llama32_3b",
        "site": "down_input",
        "layer": 1,
        "range_law_points": int(len(law)),
        "point_counts": {str(k): int(v) for k, v in law.source_family.value_counts().items()},
        "font_family_resolved": resolved_serif_family(),
        **fit,
    }
    (here / "fig3_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
