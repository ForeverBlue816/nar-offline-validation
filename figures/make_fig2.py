#!/usr/bin/env python3
"""Render Figure 2 as three final-size standalone matplotlib panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from figure_style import PALETTE, clean_2d_axis, configure_style, resolved_serif_family, save_panel

MODEL = "llama32_3b"
SITE = "down"
LAYERS = 28
GROUP = 128


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    subset = data[data.model.eq(MODEL) & data.site.eq(SITE)].copy()
    expected = set(range(LAYERS))
    for method in ("hadamard", "duquant_style", "nar"):
        found = set(subset[subset.method.eq(method)].layer.astype(int))
        if found != expected:
            raise AssertionError(f"{method}: incomplete Figure 2 layers {sorted(expected - found)}")
    paired = subset[subset.method.isin(["hadamard", "nar"])]
    if paired[["mean_group_range", "nmse"]].isna().any().any():
        raise AssertionError("Figure 2 contains blank range or NMSE values")
    return subset


def paired_metric(data: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pivot = data[data.method.isin(["hadamard", "nar"])].pivot(
        index="layer", columns="method", values=column
    ).sort_index()
    reduction = 100.0 * (pivot.hadamard - pivot.nar) / pivot.hadamard
    return (
        pivot.index.to_numpy(),
        pivot.hadamard.to_numpy(),
        pivot.nar.to_numpy(),
        float(reduction.mean()),
    )


def new_panel() -> tuple[plt.Figure, plt.Axes]:
    configure_style()
    fig, ax = plt.subplots(figsize=(1.85, 1.72))
    fig.subplots_adjust(left=0.28, right=0.96, bottom=0.25, top=0.95)
    clean_2d_axis(ax)
    ax.tick_params(labelsize=7.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    return fig, ax


def render_a(data: pd.DataFrame, outbase: Path) -> None:
    fig, ax = new_panel()
    reference = 1.0 / GROUP
    ax.axhline(reference, color=PALETTE["reference"], lw=0.65, ls=(0, (3, 2)), zorder=0)
    specs = (
        ("hadamard", "Hadamard", PALETTE["hadamard"], 1.0, 2.2),
        ("duquant_style", "DuQuant", PALETTE["duquant"], 1.05, 2.2),
        ("nar", "PrismQuant k=max", PALETTE["prismquant"], 1.8, 2.7),
    )
    for method, label, color, width, size in specs:
        part = data[data.method.eq(method)].sort_values("layer")
        ax.plot(
            part.layer, part.f, color=color, lw=width, marker="o", ms=size,
            mec="white", mew=0.2, label=label,
        )
    ax.set_xlim(-0.8, LAYERS - 0.2)
    ax.set_xticks([0, LAYERS - 1])
    ax.set_xlabel("layer index")
    ax.set_ylabel("null-space energy fraction, f")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    legend = ax.legend(
        loc="upper right", fontsize=7.0, handlelength=1.05, handletextpad=0.35,
        labelspacing=0.18, borderaxespad=0.15, frameon=False,
    )
    for text, color in zip(
        legend.get_texts(),
        [PALETTE["hadamard"], PALETTE["duquant"], PALETTE["prismquant"]],
    ):
        text.set_color(color)
    save_panel(fig, outbase)


def render_metric(data: pd.DataFrame, column: str, ylabel: str, outbase: Path) -> float:
    fig, ax = new_panel()
    fig.subplots_adjust(left=0.28, right=0.96, bottom=0.25, top=0.84)
    layer, hadamard, prism, reduction = paired_metric(data, column)
    ax.plot(layer, hadamard, color=PALETTE["hadamard"], lw=1.0, marker="o", ms=2.2)
    ax.plot(
        layer, prism, color=PALETTE["prismquant"], lw=1.8, marker="o", ms=2.7,
        mec="white", mew=0.2,
    )
    ax.set_xlim(-0.8, LAYERS - 0.2)
    ax.set_xticks([0, LAYERS - 1])
    ax.set_xlabel("layer index")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if column == "mean_group_range" else "%.3f"))
    fig.text(
        0.96, 0.955, f"mean reduction {reduction:.1f}%",
        fontsize=7.5, color=PALETTE["text"], ha="right", va="top",
    )
    save_panel(fig, outbase)
    return reduction


def make_preview(here: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(5.55, 1.72))
    for ax, letter in zip(axes, "abc"):
        ax.imshow(plt.imread(here / f"fig2{letter}.png"))
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01)
    fig.savefig(here / "fig2_preview.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "fig2_capture.csv")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    data = validate_data(pd.read_csv(args.csv))
    render_a(data, here / "fig2a")
    range_reduction = render_metric(data, "mean_group_range", "mean group range", here / "fig2b")
    nmse_reduction = render_metric(data, "nmse", "activation NMSE", here / "fig2c")
    make_preview(here)
    metadata = {
        "model": MODEL,
        "site": SITE,
        "layers": LAYERS,
        "group_size": GROUP,
        "mean_range_reduction_percent": range_reduction,
        "mean_nmse_reduction_percent": nmse_reduction,
        "font_family_resolved": resolved_serif_family(),
        "duquant_display_label": "DuQuant",
        "reference_line": "G/d retained but unlabeled",
        "panel_b_secondary_series": "removed; it encoded reduction rather than raw range",
    }
    (here / "fig2_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
