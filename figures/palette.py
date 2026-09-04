"""Shared, auditable method palettes for Figures 1--3."""

from __future__ import annotations

from matplotlib import font_manager


SERIF_CANDIDATES = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]


PALETTES = {
    "A": {
        "identity": "#7F7F7F",
        "hadamard": "#E69F00",
        "duquant_style": "#009E73",
        "prismquant": "#0072B2",
    },
    "B": {
        "identity": "#858484",
        "hadamard": "#FFE39C",
        "duquant_style": "#8DBEA6",
        "prismquant": "#001638",
    },
}

# Variant B is computed from Variant A in CAM02-UCS. Supporting-series
# lightness is raised while PrismQuant is darkened; the unusually wide
# lightness spacing is required by the >=15 deltaE grayscale gate.
MUTED_CAM02_TARGET_J = {
    "identity": 58.0,
    "hadamard": 94.0,
    "duquant_style": 75.0,
    "prismquant": 10.0,
}
MUTED_CHROMA_SCALE = {
    "identity": 0.60,
    "hadamard": 0.60,
    "duquant_style": 0.60,
    "prismquant": 0.70,
}


def get_palette(name: str) -> dict[str, str]:
    try:
        return dict(PALETTES[name.upper()])
    except KeyError as exc:
        raise ValueError(f"unknown palette variant {name!r}; choose A or B") from exc


def resolved_serif_family() -> str:
    """Return and validate the family Matplotlib resolves from the required list."""
    path = font_manager.findfont(font_manager.FontProperties(family=SERIF_CANDIDATES))
    family = font_manager.FontProperties(fname=path).get_name()
    if "Sans" in family:
        raise RuntimeError(f"required serif stack resolved to a sans-serif font: {family} ({path})")
    return family
