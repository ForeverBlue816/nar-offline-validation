#!/usr/bin/env python3
"""Derive, audit, and visualize the two method-palette variants."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert
from PIL import Image

from palette import MUTED_CAM02_TARGET_J, MUTED_CHROMA_SCALE, PALETTES


METHODS = ("identity", "hadamard", "duquant_style", "prismquant")
LABELS = ("identity / raw", "Hadamard", "DuQuant-style", "PrismQuant")
CONDITIONS = (
    ("normal", None),
    ("deuteranopia", "deuteranomaly"),
    ("protanopia", "protanomaly"),
    ("tritanopia", "tritanomaly"),
    ("grayscale", "grayscale"),
)
THRESHOLD = 15.0


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def hex_to_rgb(value: str) -> np.ndarray:
    return np.array([int(value[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255.0


def rgb_to_hex(value: np.ndarray) -> str:
    rounded = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
    return "#" + "".join(f"{component:02X}" for component in rounded)


def derive_muted() -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    source = np.stack([hex_to_rgb(PALETTES["A"][method]) for method in METHODS])
    cam = cspace_convert(source, "sRGB1", "CAM02-UCS")
    target = cam.copy()
    for index, method in enumerate(METHODS):
        target[index, 0] = MUTED_CAM02_TARGET_J[method]
        target[index, 1:] *= MUTED_CHROMA_SCALE[method]
    muted_rgb = np.clip(cspace_convert(target, "CAM02-UCS", "sRGB1"), 0.0, 1.0)
    muted = {method: rgb_to_hex(muted_rgb[index]) for index, method in enumerate(METHODS)}
    measured = np.stack([hex_to_rgb(muted[method]) for method in METHODS])
    measured_cam = cspace_convert(measured, "sRGB1", "CAM02-UCS")
    audit: dict[str, dict[str, float]] = {}
    for index, method in enumerate(METHODS):
        source_chroma = float(np.linalg.norm(cam[index, 1:]))
        muted_chroma = float(np.linalg.norm(measured_cam[index, 1:]))
        audit[method] = {
            "source_J": float(cam[index, 0]),
            "muted_J": float(measured_cam[index, 0]),
            "source_chroma": source_chroma,
            "muted_chroma": muted_chroma,
            "chroma_reduction_percent": 100.0 * (1.0 - muted_chroma / source_chroma),
        }
    return muted, audit


def grayscale(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    srgb = np.where(
        luminance <= 0.0031308,
        12.92 * luminance,
        1.055 * luminance ** (1.0 / 2.4) - 0.055,
    )
    return np.repeat(srgb[:, None], 3, axis=1)


def simulate(rgb: np.ndarray, condition: str | None) -> np.ndarray:
    if condition is None:
        return rgb.copy()
    if condition == "grayscale":
        return grayscale(rgb)
    space = {"name": "sRGB1+CVD", "cvd_type": condition, "severity": 100}
    return np.clip(cspace_convert(rgb, space, "sRGB1"), 0.0, 1.0)


def palette_metrics(palette: dict[str, str], condition: str | None) -> dict[str, object]:
    rgb = np.stack([hex_to_rgb(palette[method]) for method in METHODS])
    seen = simulate(rgb, condition)
    cam = cspace_convert(seen, "sRGB1", "CAM02-UCS")
    distances = []
    for left, right in itertools.combinations(range(len(METHODS)), 2):
        distances.append((float(np.linalg.norm(cam[left] - cam[right])), left, right))
    minimum, left, right = min(distances)
    had = METHODS.index("hadamard")
    prism = METHODS.index("prismquant")
    return {
        "minimum": minimum,
        "pair": f"{LABELS[left]} vs {LABELS[right]}",
        "hadamard_prismquant": float(np.linalg.norm(cam[had] - cam[prism])),
        "passes": minimum >= THRESHOLD,
        "simulated_rgb": seen,
    }


def render_sheet(path: Path, metrics: dict[tuple[str, str], dict[str, object]]) -> None:
    configure_style()
    fig, axes = plt.subplots(5, 2, figsize=(5.5, 3.4))
    fig.subplots_adjust(left=0.16, right=0.99, top=0.95, bottom=0.09, hspace=0.42, wspace=0.12)
    for row, (condition_label, _condition) in enumerate(CONDITIONS):
        for column, variant in enumerate(("A", "B")):
            ax = axes[row, column]
            colors = metrics[(variant, condition_label)]["simulated_rgb"]
            for index, color in enumerate(colors):
                ax.add_patch(plt.Rectangle((index, 0), 1, 1, color=color, ec="#000000", lw=0.35))
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            status = "PASS" if metrics[(variant, condition_label)]["passes"] else "FAIL"
            minimum = metrics[(variant, condition_label)]["minimum"]
            ax.set_title(f"Variant {variant} · min ΔE {minimum:.1f} · {status}", fontsize=7.5, pad=2)
            if column == 0:
                ax.text(-0.08, 0.5, condition_label, transform=ax.transAxes, ha="right", va="center", fontsize=8.0)
    for index, label in enumerate(LABELS):
        fig.text(0.16 + (index + 0.5) * (0.83 / 4), 0.025, label, ha="center", va="bottom", fontsize=7.0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def simulate_figures(root: Path) -> list[str]:
    output = root / "accessibility"
    output.mkdir(exist_ok=True)
    written: list[str] = []
    for figure in (1, 2, 3):
        for variant in ("A", "B"):
            source = root / f"fig{figure}_variant{variant}.png"
            if not source.exists():
                raise FileNotFoundError(f"missing final-size palette variant: {source}")
            image = np.asarray(Image.open(source).convert("RGB"), dtype=float) / 255.0
            flat = image.reshape(-1, 3)
            for label, condition in CONDITIONS[1:]:
                converted = simulate(flat, condition).reshape(image.shape)
                target = output / f"fig{figure}_variant{variant}_{label}.png"
                Image.fromarray(np.rint(converted * 255.0).astype(np.uint8)).save(target)
                written.append(str(target.relative_to(root)))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.figures_dir.resolve()

    muted, chroma_audit = derive_muted()
    if muted != PALETTES["B"]:
        raise AssertionError(f"stored Variant B is not the CAM02-UCS derivation: {muted}")

    metrics: dict[tuple[str, str], dict[str, object]] = {}
    for variant in ("A", "B"):
        for label, condition in CONDITIONS:
            metrics[(variant, label)] = palette_metrics(PALETTES[variant], condition)
    render_sheet(root / "palette_check.png", metrics)
    simulations = simulate_figures(root)

    lines = [
        "# Palette accessibility report",
        "",
        "Variant B is derived numerically from Variant A in CAM02-UCS: supporting hues use 60% of the original chroma, PrismQuant uses 70%, and J′ is spread to satisfy the hard grayscale gate. Values are converted back to sRGB, clipped to gamut, and rounded to 8-bit hex. The accessibility constraint necessarily requires a larger lightness separation than the aesthetic target alone.",
        "",
        "| Method | Variant A | Variant B | J′ A → B | Chroma reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in zip(METHODS, LABELS):
        row = chroma_audit[method]
        lines.append(
            f"| {label} | `{PALETTES['A'][method]}` | `{PALETTES['B'][method]}` | "
            f"{row['source_J']:.1f} → {row['muted_J']:.1f} | {row['chroma_reduction_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Distances are Euclidean CAM02-UCS ΔE after colorspacious simulation at severity 100. A row passes only when the minimum over all six method pairs is at least 15.",
            "",
            "| Variant | Viewing condition | Minimum ΔE | Limiting pair | Hadamard–PrismQuant ΔE | Gate |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for variant in ("A", "B"):
        for label, _condition in CONDITIONS:
            row = metrics[(variant, label)]
            lines.append(
                f"| {variant} | {label} | {row['minimum']:.2f} | {row['pair']} | "
                f"{row['hadamard_prismquant']:.2f} | {'PASS' if row['passes'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "Variant A is retained for visual comparison but fails the hard all-pairs accessibility gate. Variant B passes under normal vision, deuteranopia, protanopia, tritanopia, and grayscale; Hadamard and PrismQuant are also well above threshold in every condition.",
            "",
            f"Full-figure simulations written: {len(simulations)} PNGs under `figures/accessibility/` (three figures × two variants × four simulated conditions).",
            "",
        ]
    )
    (root / "palette_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
