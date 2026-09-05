"""Shared matplotlib style for PrismQuant manuscript panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "prismquant": "#1D3557", "prismquant_light": "#457B9D",
    "hadamard": "#A8DADC", "duquant": "#E63946",
    "identity": "#457B9D", "reference": "#1D3557", "text": "#1D3557",
    "zero": "#F1FAEE", "pane_edge": "#C9D6DF", "grid": "#DCE4EA",
}
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "prismquant_height", ["#F1FAEE", "#A8DADC", "#457B9D", "#1D3557"])


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.5,
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
    ax.tick_params(labelsize=6.0)


def save_panel(fig: plt.Figure, outbase: Path, dpi: int = 600, **alignment) -> None:
    """Exact physical dimensions, editable vector text, render-time alignment."""
    from audit_panel_alignment import require_matplotlib_panel_alignment
    qa = outbase.parent / "qa"
    qa.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(
        fig, json_out=qa / (outbase.name + ".alignment.json"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **alignment)
    fig.savefig(outbase.with_suffix(".svg"), dpi=dpi)
    svg_path = outbase.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=dpi)
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi)
    plt.close(fig)
