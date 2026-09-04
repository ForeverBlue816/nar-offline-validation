"""Shared matplotlib style for PrismQuant manuscript panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "prismquant": "#004C94",
    "prismquant_light": "#3C93FA",
    "hadamard": "#73CC80",
    "duquant": "#0FA69D",
    "identity": "#52647A",
    "reference": "#2D3F54",
    "text": "#2D3F54",
    "zero": "#F2F4F6",
    "pane_edge": "#D0D5DB",
}

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "prismquant_sequential",
    [PALETTE["zero"], PALETTE["prismquant_light"], PALETTE["prismquant"]],
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "prismquant_diverging",
    [PALETTE["hadamard"], PALETTE["zero"], PALETTE["prismquant"]],
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "axes.edgecolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def resolved_serif_family() -> str:
    prop = font_manager.FontProperties(
        family=["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    )
    path = font_manager.findfont(prop)
    family = font_manager.FontProperties(fname=path).get_name()
    if "Sans" in family:
        raise RuntimeError(f"serif request resolved to sans: {family} ({path})")
    return family


def clean_2d_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.0)


def save_panel(fig: plt.Figure, outbase: Path, dpi: int = 300) -> None:
    """Save at the figure's exact physical dimensions; never post-scale."""
    fig.savefig(outbase.with_suffix(".svg"))
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi)
    plt.close(fig)
