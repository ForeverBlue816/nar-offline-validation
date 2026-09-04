#!/usr/bin/env python3
"""Render Figure 3 from BOS-excluded geometry and frozen range diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from palette import get_palette, resolved_serif_family


BLACK = "#000000"
GROUP = 128
SLOTS = 64


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def import_alignment_helper() -> Any:
    helper = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(helper))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def panel_letter(ax: plt.Axes, letter: str) -> None:
    offset = mpl.transforms.ScaledTranslation(-8 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + offset, fontsize=9.0,
            fontweight="bold", color=BLACK, va="bottom", ha="right")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def range_law(repo: Path, output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    e1c_path = repo / "results" / "llama32_3b" / "e1c_range_vs_k.csv"
    e1c = pd.read_csv(e1c_path)
    for row in e1c.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "source_family": "E1c activations",
                "source_artifact": "results/llama32_3b/e1c_range_vs_k.csv",
                "site": row.site,
                "layer": int(row.layer),
                "group_size": int(row.b),
                "configuration": f"nar_k{int(row.k)}",
                "method": "nar",
                "absorbed_energy_fraction": float(row.absorbed_energy_fraction),
                "sqrt_one_minus_f": float(row.sqrt_one_minus_absorbed_energy_fraction),
                "range_ratio_vs_hadamard": 1.0 - float(row.range_reduction_vs_hadamard),
                "range_ratio_definition": "mean_group_range / paired Hadamard mean_group_range at identical model, site, layer, and group_size",
            }
        )

    e7_path = repo / "results" / "llama32_3b" / "e7_range_vs_k.csv"
    e7 = pd.read_csv(e7_path)
    for row in e7.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "source_family": "E7 V cache",
                "source_artifact": "results/llama32_3b/e7_range_vs_k.csv",
                "site": "value_cache",
                "layer": int(row.layer),
                "group_size": int(row.b),
                "configuration": f"nar_k{int(row.k)}",
                "method": "nar",
                "absorbed_energy_fraction": float(row.absorbed_energy_fraction),
                "sqrt_one_minus_f": float(row.sqrt_one_minus_absorbed_energy_fraction),
                "range_ratio_vs_hadamard": 1.0 - float(row.range_reduction_vs_hadamard),
                "range_ratio_definition": "mean_group_range / paired Hadamard mean_group_range at identical model, site, layer, and group_size",
            }
        )

    e20_path = repo / "results" / "llama32_3b" / "e20_range_vs_config.csv"
    e20 = pd.read_csv(e20_path)
    e20 = e20[e20.row.str.startswith("nar_")]
    for row in e20.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "source_family": "E20 multi-slot",
                "source_artifact": "results/llama32_3b/e20_range_vs_config.csv",
                "site": row.site,
                "layer": int(row.layer),
                "group_size": int(row.group),
                "configuration": row.row,
                "method": "nar",
                "absorbed_energy_fraction": float(row.absorbed_energy_fraction),
                "sqrt_one_minus_f": float(row.sqrt_one_minus_f),
                "range_ratio_vs_hadamard": float(row.range_ratio_vs_hadamard),
                "range_ratio_definition": "mean_group_range / paired Hadamard mean_group_range at identical model, site, layer, and group_size",
            }
        )

    frame = pd.DataFrame(rows)
    numeric = frame[["sqrt_one_minus_f", "range_ratio_vs_hadamard"]]
    if not np.isfinite(numeric.to_numpy()).all():
        raise AssertionError("Figure 3 range law contains a non-finite value")
    if not frame.model.eq("llama32_3b").all():
        raise AssertionError("main Figure 3 must not mix models")
    frame.to_csv(output, index=False)
    return frame


def render(
    projections: pd.DataFrame,
    eigenspace: pd.DataFrame,
    law: pd.DataFrame,
    geometry: dict[str, Any],
    outbase: Path,
    palette_name: str,
) -> dict[str, float]:
    configure_style()
    palette = get_palette(palette_name)
    raw_color = palette["identity"]
    had_color = palette["hadamard"]
    prism_color = palette["prismquant"]

    fig, axes = plt.subplots(1, 3, figsize=(5.50, 2.55))
    fig.subplots_adjust(left=0.092, right=0.985, bottom=0.235, top=0.87, wspace=0.37)

    ax = axes[0]
    p1 = projections.projection_v1.to_numpy()
    p2 = projections.projection_v2.to_numpy()
    p1 = (p1 - p1.mean()) / p1.std(ddof=0)
    p2 = (p2 - p2.mean()) / p2.std(ddof=0)
    ax.scatter(p1, p2, s=2.0, color=raw_color, alpha=0.18, linewidths=0, rasterized=True)
    arrow_scale = 1.8
    had_arrow = geometry["arrows"]["hadamard"]
    had_dx = arrow_scale * float(had_arrow["projection_v1"])
    had_dy = arrow_scale * float(had_arrow["projection_v2"])
    ax.scatter([had_dx], [had_dy], s=16, facecolor=had_color, edgecolor=BLACK,
               linewidth=0.35, zorder=5)
    ax.text(-3.0, -1.25, f"Hadamard\n{had_arrow['in_plane_length']:.3f}",
            color=had_color, fontsize=7.5, ha="left", va="top", linespacing=0.9)
    prism_arrow = geometry["arrows"]["nar"]
    prism_dx = arrow_scale * float(prism_arrow["projection_v1"])
    prism_dy = arrow_scale * float(prism_arrow["projection_v2"])
    if prism_dx < 0:
        prism_dx, prism_dy = -prism_dx, -prism_dy
    ax.annotate("", xy=(prism_dx, prism_dy), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": prism_color, "lw": 2.0,
                            "mutation_scale": 7.5})
    ax.text(0.35, 0.75, f"PrismQuant\n{prism_arrow['in_plane_length']:.3f}",
            color=prism_color, fontsize=7.5, ha="left", va="bottom", linespacing=0.9)
    limit = max(3.0, float(np.quantile(np.abs(np.concatenate([p1, p2])), 0.995)))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="#B0B0B0", lw=0.45, zorder=0)
    ax.axvline(0, color="#B0B0B0", lw=0.45, zorder=0)
    ax.set_xlabel("token projection on v₁ (s.d.)")
    ax.set_ylabel("token projection on v₂ (s.d.)")

    clean_axis(ax)
    panel_letter(ax, "a")

    ax = axes[1]
    line_specs = (
        (1, "layer 1", prism_color, "-", 1.9),
        (13, "layer 13", "#555555", (0, (3, 2)), 0.9),
        (27, "layer 27", "#999999", (0, (1, 2)), 0.9),
    )
    for layer, label, color, linestyle, width in line_specs:
        part = eigenspace[eigenspace.layer.eq(layer)].sort_values("rank")
        if len(part) != 256:
            raise AssertionError(f"layer {layer}: rank-256 eigenspectrum missing")
        ax.plot(part["rank"], part.cumulative_fraction_total_energy, color=color,
                ls=linestyle, lw=width, label=label)
    ax.axvline(SLOTS, color="#B0B0B0", lw=0.75, ls=(0, (3, 2)))
    ax.text(72, 0.94, "64-slot\nlimit", color=BLACK,
            fontsize=7.5, ha="left", va="top", linespacing=0.9)
    ax.set_xlim(1, 256)
    ax.set_xticks([1, 64, 256])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("eigen-directions retained, k")
    ax.set_ylabel("cumulative activation energy")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    endpoints = {layer: float(eigenspace[eigenspace.layer.eq(layer)].sort_values("rank").cumulative_fraction_total_energy.iloc[-1])
                 for layer in (1, 13, 27)}
    ax.text(250, endpoints[27] + 0.055, "layer 27", color="#999999", fontsize=7.5,
            ha="right", va="bottom")
    ax.text(250, endpoints[13] + 0.045, "layer 13", color="#555555", fontsize=7.5,
            ha="right", va="bottom")
    ax.text(250, endpoints[1] - 0.045, "layer 1", color=prism_color, fontsize=7.5,
            ha="right", va="top")
    clean_axis(ax)
    panel_letter(ax, "b")

    ax = axes[2]
    marker_specs = {
        "E1c activations": ("o", 9.0, 0.28),
        "E7 V cache": ("s", 11.0, 0.42),
        "E20 multi-slot": ("^", 12.0, 0.52),
    }
    for family, (marker, size, alpha) in marker_specs.items():
        part = law[law.source_family.eq(family)]
        ax.scatter(part.sqrt_one_minus_f, part.range_ratio_vs_hadamard,
                   color=prism_color, marker=marker, s=size, alpha=alpha,
                   edgecolors="none", label=family, rasterized=True)
    x = law.sqrt_one_minus_f.to_numpy()
    y = law.range_ratio_vs_hadamard.to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = intercept + slope * x
    r_squared = 1.0 - float(np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2))
    grid = np.linspace(max(0, x.min() - 0.02), min(1.02, x.max() + 0.02), 100)
    ax.plot(grid, intercept + slope * grid, color=prism_color, lw=2.0)
    ax.plot([0, 1], [0, 1], color=BLACK, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.04, 0.96, f"pooled fit  R² = {r_squared:.2f}", transform=ax.transAxes,
            color=BLACK, fontsize=7.5, ha="left", va="top")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(1.02, float(np.quantile(y, 0.995)) * 1.05))
    ax.set_xlabel("predicted scale, √(1 − f)")
    ax.set_ylabel("range / paired Hadamard range")
    ax.legend(loc="lower right", handletextpad=0.35, labelspacing=0.25)
    clean_axis(ax)
    panel_letter(ax, "c")

    fig.text(0.985, 0.025, "Llama-3.2-3B · down-input geometry · BOS excluded",
             fontsize=7.0, color=BLACK, ha="right", va="bottom")
    fig.canvas.draw()
    require_matplotlib_panel_alignment = import_alignment_helper()
    require_matplotlib_panel_alignment(
        fig, axes=list(axes), panel_ids=["a", "b", "c"],
        row_groups=[{"id": "geometry-law", "panels": ["a", "b", "c"]}],
        require_panel_labels=True,
        json_out=outbase.with_suffix(".alignment.json"),
        overlay_svg=outbase.with_suffix(".alignment-overlay.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(outbase.with_suffix(".svg"))
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return {"fit_intercept": float(intercept), "fit_slope": float(slope), "fit_r_squared": r_squared}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    here = Path(__file__).resolve().parent
    projections = pd.read_csv(here / "fig3_token_projections.csv")
    eigenspace = pd.read_csv(here / "fig3_eigenspace_r256.csv")
    geometry = json.loads((here / "fig3_geometry_metadata.json").read_text())
    law = range_law(repo, here / "fig3_range_law.csv")

    metadata: dict[str, Any] = {
        "model": "llama32_3b",
        "site": "down_input",
        "layer": 1,
        "resolved_font_family": resolved_serif_family(),
        "naming": "paper label PrismQuant; CSV method value nar retained",
        "geometry": geometry,
        "range_ratio_definition": law.range_ratio_definition.iloc[0],
        "range_law_points": int(len(law)),
        "figures": {},
    }
    for palette_name in ("A", "B"):
        metadata["figures"][f"variant{palette_name}"] = render(
            projections, eigenspace, law, geometry, here / f"fig3_variant{palette_name}", palette_name
        )
    for suffix in (".svg", ".pdf", ".png", ".alignment.json", ".alignment-overlay.svg"):
        shutil.copyfile(here / f"fig3_variantA{suffix}", here / f"fig3_geometry{suffix}")
    (here / "fig3_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    caption = (
        "PrismQuant makes leading activation directions available to the affine quantizer's null space. "
        "a, Non-BOS calibration tokens projected onto the two leading BOS-excluded second-moment directions; "
        "arrows show the matched Hadamard and PrismQuant preimages of one groupwise null-space direction. "
        "b, Rank-256 cumulative spectra for three down-input layers; the dashed line marks 64 group-128 "
        "null-space slots. c, Measured range ratios across E1c activations, E7 V cache and E20 multi-slot "
        "diagnostics versus the square-root residual-energy predictor. Ratios use the paired Hadamard row at "
        "the identical model, site, layer and group size."
    )
    (here / "fig3_caption.txt").write_text(caption + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
