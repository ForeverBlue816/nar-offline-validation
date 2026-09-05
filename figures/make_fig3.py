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
    fig, ax = plt.subplots(figsize=(2.65, 2.35))
    fig.subplots_adjust(left=left, right=0.94, bottom=0.24, top=0.94)
    clean_2d_axis(ax)
    ax.tick_params(labelsize=6.0)
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
    scale = 1.0
    had = geometry["arrows"]["hadamard"]
    had_x = scale * float(had["projection_v1"])
    had_y = scale * float(had["projection_v2"])
    ax.annotate(
        "", xy=(had_x, had_y), xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["hadamard"], "lw": 1.0, "mutation_scale": 3.5, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(0.04, 0.96, f"Hadamard\n{had['in_plane_length']:.3f}", color=PALETTE["hadamard"],
            fontsize=7.0, ha="left", va="top", linespacing=1.2, transform=ax.transAxes)
    prism = geometry["arrows"]["nar"]
    prism_x = scale * float(prism["projection_v1"])
    prism_y = scale * float(prism["projection_v2"])
    if prism_x < 0:
        prism_x, prism_y = -prism_x, -prism_y
    ax.annotate(
        "", xy=(prism_x, prism_y), xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": PALETTE["prismquant"], "lw": 1.8, "mutation_scale": 7.0, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(0.62, 0.96, f"PrismQuant\n{prism['in_plane_length']:.2f}", color=PALETTE["prismquant"],
            fontsize=7.0, ha="left", va="top", linespacing=1.2, transform=ax.transAxes)
    cloud = np.column_stack([p1, p2])
    quantiles = np.quantile(cloud, [0.005, 0.995], axis=0)
    low = np.minimum(quantiles[0], cloud.min(0))
    high = np.maximum(quantiles[1], cloud.max(0))
    pad = 0.08 * (high - low)
    limits = np.column_stack([low - pad, high + pad])
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
    geometry["cloud"] = {
        "initial_percentile_limits": quantiles.tolist(), "axis_limits": limits.tolist(),
        "rows": len(cloud), "inside_frame": int(np.all((cloud >= limits[:, 0]) & (cloud <= limits[:, 1]), axis=1).sum()),
        "projection_mean": projections[["projection_v1", "projection_v2"]].mean().tolist(),
        "projection_sd_ddof0": projections[["projection_v1", "projection_v2"]].std(ddof=0).tolist(),
        "bounds_rule": "0.5-99.5 percentiles expanded to all observations, plus 8% padding",
        "arrow_scale": 1.0,
    }
    ax.axhline(0, color=PALETTE["pane_edge"], lw=0.45, zorder=0)
    ax.axvline(0, color=PALETTE["pane_edge"], lw=0.45, zorder=0)
    ax.set_xlabel("projection on v1 (s.d.)", fontsize=7.0)
    ax.set_ylabel("projection on v2 (s.d.)", fontsize=7.0)
    save_panel(fig, outbase)


def render_b(eigenspace: pd.DataFrame, outbase: Path) -> None:
    fig, ax = new_panel(left=0.29)
    fig.subplots_adjust(right=0.76)
    specs = (
        (1, PALETTE["hadamard"], "-", 1.0, "layer 1"),
        (13, PALETTE["hadamard"], (0, (3, 2)), 1.0, "layer 13"),
        (27, PALETTE["prismquant"], "-", 1.2, "layer 27"),
    )
    endpoints: dict[int, float] = {}
    for layer, color, linestyle, width, _label in specs:
        part = eigenspace[eigenspace.layer.eq(layer)].sort_values("rank")
        if len(part) != 256:
            raise AssertionError(f"layer {layer}: rank-256 eigenspectrum missing")
        ax.plot(part["rank"], part.cumulative_fraction_total_energy, color=color, ls=linestyle, lw=width)
        endpoints[layer] = float(part.cumulative_fraction_total_energy.iloc[-1])
    ax.set_xscale("log")
    ax.set_xlim(1, 256)
    ax.minorticks_off()
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1, 4, 16, 64, 256])
    ax.set_xticklabels(["1", "4", "16", "64", "256"])
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    ax.set_xlabel("directions retained, k", fontsize=7.0)
    ax.set_ylabel("cumulative energy", fontsize=7.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    for layer, offset, color in [(27, 0, PALETTE["prismquant"]),
                                  (13, 5, PALETTE["hadamard"]),
                                  (1, -5, PALETTE["hadamard"])]:
        ax.annotate(f"layer {layer}", xy=(256, endpoints[layer]), xytext=(5, offset),
                    textcoords="offset points", color=color, fontsize=7,
                    ha="left", va="center", annotation_clip=False)
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
        "E1c activations": (PALETTE["prismquant"], "o", 7.0, "E1c activations"),
        "E7 V cache": (PALETTE["identity"], "s", 7.0, "E7 V cache"),
        "E20 multi-slot": (PALETTE["hadamard"], "^", 7.0, "E20 multi-slot"),
    }
    for family, (color, marker, size, label) in specs.items():
        part = law[law.source_family.eq(family)]
        if part.empty:
            continue
        if inset:
            part = part[part.sqrt_one_minus_f.between(0.85, 1.0) & part.range_ratio_vs_hadamard.between(0.85, 1.0)]
        ax.scatter(
            part.sqrt_one_minus_f, part.range_ratio_vs_hadamard,
            color=color, marker=marker, s=size, alpha=0.35,
            edgecolors="none", rasterized=True, label=None if inset else label,
        )


def render_c(law: pd.DataFrame, outbase: Path, pooled: pd.DataFrame | None = None) -> dict[str, float | bool]:
    fig, ax = new_panel(left=0.30)
    scatter_sources(ax, law)
    intercept, slope, r_squared = fit_law(law if pooled is None else pooled)
    grid = np.linspace(0, 1.02, 150)
    ax.plot([0, 1], [0, 1], color=PALETTE["reference"], lw=0.8, ls=(0, (3, 2)))
    ax.plot(grid, intercept + slope * grid, color=PALETTE["reference"], lw=0.6)
    ax.text(0.035, 0.965, f"Pooled R² = {r_squared:.2f}", transform=ax.transAxes,
            color=PALETTE["text"], fontsize=7.0, ha="left", va="top")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(1.02, float((law if pooled is None else pooled).range_ratio_vs_hadamard.max()) * 1.04))
    ax.set_xlabel("√(1 − f)", fontsize=7.0)
    ax.set_ylabel("range / paired Hadamard range", fontsize=7.0)
    crowded = int(((law.sqrt_one_minus_f > 0.85) & (law.range_ratio_vs_hadamard > 0.85)).sum()) >= 100
    if crowded:
        inset = ax.inset_axes([0.18, 0.63, 0.28, 0.23])
        inset.set_facecolor("#FAFCFD")
        scatter_sources(inset, law, inset=True)
        inset.plot([0.85, 1.0], [0.85, 1.0], color=PALETTE["reference"], lw=0.55, ls=(0, (3, 2)))
        visible = (grid >= 0.85) & (grid <= 1.0) & (intercept + slope * grid >= 0.85)
        inset.plot(grid[visible], intercept + slope * grid[visible], color=PALETTE["reference"], lw=0.6)
        inset.set_xlim(0.85, 1.0)
        inset.set_ylim(0.85, 1.0)
        inset.set_xticks([0.85, 1.00])
        inset.set_yticks([0.85, 1.00])
        inset.tick_params(labelsize=6.0, length=1.5, pad=2)
        inset.get_xticklabels()[0].set_ha("left")
        for spine in inset.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color(PALETTE["pane_edge"])
        from matplotlib.patches import Rectangle, ConnectionPatch
        corner = Rectangle((0.85, 0.85), 0.15, 0.15, facecolor="none",
                           edgecolor=PALETTE["pane_edge"], linewidth=0.65, zorder=5)
        ax.add_patch(corner)
        connector = ConnectionPatch(xyA=(0.85, 1.0), coordsA=ax.transData,
                                    xyB=(1, 1), coordsB=inset.transAxes,
                                    color=PALETTE["pane_edge"], linewidth=0.55, zorder=1)
        fig.add_artist(connector)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    legend = ax.legend(
        loc="lower right", fontsize=6.5, handlelength=1.0, handletextpad=0.5,
        labelspacing=0.4, borderaxespad=0.4, frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])
    save_panel(fig, outbase)
    return {"fit_intercept": intercept, "fit_slope": slope, "fit_r_squared": r_squared, "corner_inset": crowded, "inset_limits": [0.85, 1.0], "fit_definition": "pooled ordinary least squares with intercept; all source rows included", "main_axis_limits": [list(ax.get_xlim()), list(ax.get_ylim())]}


def compose_panels(here: Path, stems: list[str], outbase: str, columns: int) -> None:
    """Compose editable originals with no added captions or resizing."""
    import pymupdf
    import xml.etree.ElementTree as ET
    from audit_panel_alignment import require_matplotlib_panel_alignment
    configure_style()
    rows = (len(stems) + columns - 1) // columns
    width, height = 2.65 * columns, 2.35 * rows
    fig = plt.figure(figsize=(width, height))
    doc = pymupdf.open(); page = doc.new_page(width=72*width, height=72*height)
    ns = "http://www.w3.org/2000/svg"; ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}svg", width=f"{width*72}pt", height=f"{height*72}pt", viewBox=f"0 0 {width*72} {height*72}")
    measured = []
    for i, stem in enumerate(stems):
        x, y = (i % columns) * 2.65, height - (i // columns + 1) * 2.35
        ax = fig.add_axes([x/width, y/height, 2.65/width, 2.35/height])
        ax.imshow(plt.imread(here / f"{stem}.png")); ax.axis("off")
        report = json.loads((here / "qa" / f"{stem}.alignment.json").read_text())
        x0,y0,x1,y1 = report["layout"]["panels"][0]["bbox_pt"]
        area = fig.add_axes([(x+x0/72)/width, (y+y0/72)/height,
                             (x1-x0)/72/width, (y1-y0)/72/height], label=stem)
        area.axis("off"); measured.append(area)
        with pymupdf.open(here / f"{stem}.pdf") as src:
            page.show_pdf_page(pymupdf.Rect(x*72, (height-y-2.35)*72, (x+2.65)*72, (height-y)*72), src, 0)
        node = ET.parse(here / f"{stem}.svg").getroot()
        raw = ET.tostring(node, encoding="unicode")
        ids = [element.attrib["id"] for element in node.iter() if "id" in element.attrib]
        for key in sorted(ids, key=len, reverse=True):
            raw = raw.replace(f'id="{key}"', f'id="{stem}_{key}"').replace(f"#{key})", f"#{stem}_{key})").replace(f'"#{key}"', f'"#{stem}_{key}"')
        node = ET.fromstring(raw); node.set("x", str(x*72)); node.set("y", str((height-y-2.35)*72)); root.append(node)
    require_matplotlib_panel_alignment(fig, json_out=here/"qa"/f"{outbase}.alignment.json",
        axes=measured, panel_ids=stems, row_groups=[["fig3c1", "fig3c2"]], column_groups=[], strict=True)
    fig.savefig(here / f"{outbase}_preview.png", dpi=300)
    fig.savefig(here / f"{outbase}.png", dpi=300)
    plt.close(fig)
    doc.save(here / f"{outbase}.pdf", deflate=True); doc.close()
    ET.ElementTree(root).write(here / f"{outbase}.svg", encoding="unicode", xml_declaration=True)


def make_preview(here: Path) -> None:
    compose_panels(here, ["fig3c1", "fig3c2"], "fig3c", 2)
    compose_panels(here, ["fig3a", "fig3b", "fig3c1", "fig3c2"], "fig3", 2)


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
    activation = law[law.source_family.eq("E1c activations")]
    cache_and_multislot = law[~law.source_family.eq("E1c activations")]
    fit = render_c(activation, here / "fig3c1", pooled=law)
    other_fit = render_c(cache_and_multislot, here / "fig3c2", pooled=law)
    make_preview(here)
    metadata = {
        "model": "llama32_3b",
        "site": "down_input",
        "layer": int(geometry["layer"]),
        "restored_from_commit": "721f253",
        "geometry": geometry,
        "panels": ["a", "b", "c"],
        "palette": PALETTE,
        "panel_size_inches": [2.65, 2.35],
        "range_law_points": int(len(law)),
        "point_counts": {str(k): int(v) for k, v in law.source_family.value_counts().items()},
        "font_family_resolved": resolved_serif_family(),
        **fit,
        "range_law_subpanels": {"fig3c1": {"families": ["E1c activations"], "points": len(activation), "inset": fit["corner_inset"]}, "fig3c2": {"families": ["E7 V cache", "E20 multi-slot"], "points": len(cache_and_multislot), "inset": other_fit["corner_inset"]}},
        "fit_display": "same pooled fit in both views; not a per-family fit",
    }
    (here / "fig3_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
