#!/usr/bin/env python3
"""Render Figure 2's layer-wise null-space → range → INT4-error chain."""

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
MODELS = {
    "llama32_3b": {
        "label": "Llama-3.2-3B",
        "layers": 28,
        "qkv": 3072,
        "down": 8192,
    },
    "llama31_8b": {
        "label": "Llama-3.1-8B",
        "layers": 32,
        "qkv": 4096,
        "down": 14336,
    },
}


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


def normalized_e1c(repo: Path, model: str) -> pd.DataFrame:
    source = repo / "results" / model / "e1c_per_layer.csv"
    data = pd.read_csv(source)
    data = data[data.method.isin(["hadamard_full", "nar_kmax"])].copy()
    data["site"] = data.site.map({"q_input": "qkv", "down_input": "down"})
    data["method"] = data.method.map({"hadamard_full": "hadamard", "nar_kmax": "nar"})
    data = data.rename(columns={"relative_quantization_error_nmse": "nmse"})
    data["source_artifact"] = f"results/{model}/e1c_per_layer.csv"
    return data[
        [
            "model",
            "site",
            "layer",
            "method",
            "mean_group_range",
            "nmse",
            "evaluation_tokens",
            "source_artifact",
        ]
    ]


def build_figure_csv(repo: Path, figure_stats: Path, csv_path: Path) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    crosscheck: dict[str, float] = {}
    for model, spec in MODELS.items():
        measured_path = figure_stats / model / "e16_null_space_energy_per_layer.csv"
        if not measured_path.exists():
            raise FileNotFoundError(f"missing measured Figure 2 statistics: {measured_path}")
        measured = pd.read_csv(measured_path)
        required = {"model", "site", "layer", "method", "f", "rows_used", "slots", "d"}
        if not required.issubset(measured.columns):
            raise ValueError(f"{measured_path} lacks {sorted(required - set(measured.columns))}")

        if model == "llama32_3b":
            quantitative = normalized_e1c(repo, model)
            frozen = quantitative.rename(
                columns={
                    "mean_group_range": "frozen_range",
                    "nmse": "frozen_nmse",
                    "evaluation_tokens": "frozen_rows",
                }
            )
            measured = measured.merge(
                frozen[["site", "layer", "method", "frozen_range", "frozen_nmse", "frozen_rows"]],
                on=["site", "layer", "method"],
                how="left",
            )
            for column, frozen_column in (
                ("mean_group_range", "frozen_range"),
                ("nmse", "frozen_nmse"),
            ):
                if column in measured:
                    valid = measured.method.isin(["hadamard", "nar"])
                    difference = np.abs(
                        measured.loc[valid, column].to_numpy()
                        - measured.loc[valid, frozen_column].to_numpy()
                    )
                    crosscheck[f"{model}_{column}_max_abs"] = float(np.nanmax(difference))
                measured.loc[measured.method.isin(["hadamard", "nar"]), column] = measured.loc[
                    measured.method.isin(["hadamard", "nar"]), frozen_column
                ]
            measured["evaluation_tokens"] = measured["frozen_rows"]
            measured["quantitative_source"] = f"results/{model}/e1c_per_layer.csv"
        else:
            for column in ("mean_group_range", "nmse"):
                if column not in measured.columns:
                    raise ValueError(f"{measured_path} is missing real 8B {column}")
            measured["evaluation_tokens"] = measured["rows_used"]
            measured["quantitative_source"] = "figure_stats_v2 measured replay"

        measured["null_space_source"] = "figure_stats_v2 measured replay"
        expected_layers = set(range(int(spec["layers"])))
        for site in ("qkv", "down"):
            for method in ("hadamard", "duquant_style", "nar"):
                found = set(
                    measured[
                        measured.site.eq(site) & measured.method.eq(method)
                    ].layer.astype(int)
                )
                if found != expected_layers:
                    raise AssertionError(
                        f"{model}/{site}/{method}: incomplete layers "
                        f"{sorted(expected_layers - found)}"
                    )
        quantitative_rows = measured[measured.method.isin(["hadamard", "nar"])]
        if quantitative_rows[["mean_group_range", "nmse"]].isna().any().any():
            raise AssertionError(f"{model}: blank range/NMSE values are forbidden")
        frames.append(measured)

    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values(["model", "site", "method", "layer"])
    output.to_csv(csv_path, index=False)
    return {"crosscheck": crosscheck, "rows": int(len(output))}


def panel_letter(ax: plt.Axes, letter: str) -> None:
    offset = mpl.transforms.ScaledTranslation(-8 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        letter,
        transform=ax.transAxes + offset,
        fontsize=9.0,
        fontweight="bold",
        color=BLACK,
        va="bottom",
        ha="right",
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def paired_metric(data: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = data.pivot(index="layer", columns="method", values=column).sort_index()
    if pivot[["hadamard", "nar"]].isna().any().any():
        raise AssertionError(f"blank paired {column} value")
    reduction = 100.0 * (pivot.hadamard.to_numpy() - pivot.nar.to_numpy()) / pivot.hadamard.to_numpy()
    return pivot.index.to_numpy(), pivot.hadamard.to_numpy(), pivot.nar.to_numpy(), reduction


def render_one(
    data: pd.DataFrame,
    model: str,
    site: str,
    outbase: Path,
    palette_name: str,
) -> dict[str, float]:
    configure_style()
    palette = get_palette(palette_name)
    identity_color = palette["identity"]
    had_color = palette["hadamard"]
    duq_color = palette["duquant_style"]
    prism_color = palette["prismquant"]
    spec = MODELS[model]
    subset = data[data.model.eq(model) & data.site.eq(site)].copy()
    if subset.empty:
        raise AssertionError(f"no data for {model}/{site}")

    fig, axes = plt.subplots(1, 3, figsize=(5.50, 2.45), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.235, top=0.915, wspace=0.46)
    layers = np.arange(int(spec["layers"]))

    ax = axes[0]
    reference = (int(spec[site]) // GROUP) / int(spec[site])
    ax.axhline(reference, color="#B0B0B0", lw=0.7, ls=(0, (3, 2)), zorder=0)
    methods = (
        ("hadamard", "Hadamard", had_color, 1.05, 2.2),
        ("duquant_style", "DuQuant-style", duq_color, 1.15, 2.3),
        ("nar", "PrismQuant k=max", prism_color, 1.85, 2.8),
    )
    for method, label, color, linewidth, markersize in methods:
        part = subset[subset.method.eq(method)].sort_values("layer")
        if len(part) != len(layers):
            raise AssertionError(f"{model}/{site}/{method}: incomplete f series")
        ax.plot(
            part.layer,
            part.f,
            color=color,
            lw=linewidth,
            marker="o",
            ms=markersize,
            mec="white",
            mew=0.25,
            label=label,
        )

    ax.set_ylabel("null-space energy fraction, f")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_yticks([0.0, 1.0])
    ax.legend(loc="upper right", borderaxespad=0.1, handlelength=1.4, labelspacing=0.25, labelcolor="linecolor")
    clean_axis(ax)
    panel_letter(ax, "a")

    quantitative = subset[subset.method.isin(["hadamard", "nar"])]
    range_layer, range_had, range_nar, range_reduction = paired_metric(
        quantitative, "mean_group_range"
    )
    ax = axes[1]
    ax.plot(range_layer, range_had, color=had_color, lw=1.05, marker="o", ms=2.2)
    ax.plot(range_layer, range_nar, color=prism_color, lw=1.85, marker="o", ms=2.8, mec="white", mew=0.25)
    ax.set_ylabel("mean group range")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    clean_axis(ax)
    reduction_axis = ax.twinx()
    reduction_axis.plot(
        range_layer,
        range_reduction,
        color="#666666",
        lw=0.65,
        ls=(0, (2, 2)),
        marker=".",
        ms=2.4,
        zorder=0,
    )
    reduction_axis.set_yticks([])
    reduction_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in reduction_axis.spines.values():
        spine.set_visible(False)
    reduction_axis.patch.set_visible(False)
    range_mean = float(np.mean(range_reduction))
    ax.set_title(f"mean Δrange {range_mean:.1f}%", fontsize=7.5, color=BLACK,
                 loc="left", pad=2.0)
    panel_letter(ax, "b")

    nmse_layer, nmse_had, nmse_nar, nmse_reduction = paired_metric(
        quantitative, "nmse"
    )
    ax = axes[2]
    ax.plot(nmse_layer, nmse_had, color=had_color, lw=1.05, marker="o", ms=2.2)
    ax.plot(nmse_layer, nmse_nar, color=prism_color, lw=1.85, marker="o", ms=2.8, mec="white", mew=0.25)
    ax.set_ylabel("activation NMSE")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    nmse_mean = float(np.mean(nmse_reduction))
    ax.set_title(f"mean ΔNMSE {nmse_mean:.1f}%", fontsize=7.5, color=BLACK,
                 loc="left", pad=2.0)
    clean_axis(ax)
    panel_letter(ax, "c")

    for ax in axes:
        ax.set_xlim(-0.8, int(spec["layers"]) - 0.2)
        ax.set_xticks([0, int(spec["layers"]) - 1])
        ax.set_xlabel("layer index")

    fig.text(0.105, 0.025, f"dashed: G/d = {reference:.4f}", fontsize=7.0,
             color=BLACK, ha="left", va="bottom")
    fig.text(0.985, 0.025,
             f"{spec['label']} · {'q/k/v input' if site == 'qkv' else 'down-projection input'} · group 128",
             fontsize=7.0, color=BLACK, ha="right", va="bottom")
    fig.canvas.draw()
    require_matplotlib_panel_alignment = import_alignment_helper()
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes),
        panel_ids=["a", "b", "c"],
        row_groups=[{"id": "causal-chain", "panels": ["a", "b", "c"]}],
        require_panel_labels=True,
        json_out=outbase.with_suffix(".alignment.json"),
        overlay_svg=outbase.with_suffix(".alignment-overlay.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(outbase.with_suffix(".svg"))
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return {
        "null_space_reference_G_over_d": reference,
        "mean_range_reduction_percent": range_mean,
        "mean_nmse_reduction_percent": nmse_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--figure-stats", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = here / "fig2_capture.csv"
    metadata = build_figure_csv(
        args.repo.resolve(), args.figure_stats.resolve(), csv_path
    )
    data = pd.read_csv(csv_path)
    metadata["figures"] = {}
    for palette_name in ("A", "B"):
        metadata["figures"][f"main_3b_down_variant{palette_name}"] = render_one(
            data, "llama32_3b", "down", here / f"fig2_variant{palette_name}", palette_name
        )
    for suffix in (".svg", ".pdf", ".png", ".alignment.json", ".alignment-overlay.svg"):
        shutil.copyfile(here / f"fig2_variantA{suffix}", here / f"fig2_null_space_capture{suffix}")
    appendix = {
        "appendix_3b_qkv": ("llama32_3b", "qkv", here / "fig2_null_space_capture_3b"),
        "appendix_8b_down": ("llama31_8b", "down", here / "fig2_null_space_capture_8b"),
        "appendix_8b_qkv": ("llama31_8b", "qkv", here / "fig2_null_space_capture_8b_qkv"),
    }
    for key, (model, site, outbase) in appendix.items():
        metadata["figures"][key] = render_one(data, model, site, outbase, "A")
    metadata["resolved_font_family"] = resolved_serif_family()
    metadata["naming"] = "paper label PrismQuant; CSV method value nar retained"
    (here / "fig2_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    first_sentence = (
        "Filling the null space (a) lowers the quantization range (b), "
        "which lowers the 4-bit error (c), layer by layer."
    )
    captions = {
        "fig2_caption.txt": (
            first_sentence
            + " Lines show real paired per-layer activation statistics for Llama-3.2-3B "
            "down-projection inputs. Panel a reports whole-activation energy in the "
            "groupwise quantizer null space; the dashed line marks G/d. Panel b overlays "
            "the paired layer-wise range reduction on a secondary axis. Panel c reports "
            "dynamic asymmetric group-128 INT4 NMSE."
        ),
        "fig2_appendix_caption.txt": (
            first_sentence
            + " Appendix versions repeat the same measured chain for q/k/v inputs and "
            "for Llama-3.1-8B; no value is interpolated or copied across models or sites."
        ),
    }
    for filename, caption in captions.items():
        (here / filename).write_text(caption + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
