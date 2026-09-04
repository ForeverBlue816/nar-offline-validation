#!/usr/bin/env python3
"""Build revised Figure 1 as seven final-size standalone matplotlib panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize

from figure_style import (
    PALETTE,
    SEQUENTIAL_CMAP,
    clean_2d_axis,
    configure_style,
    resolved_serif_family,
    save_panel,
)

MODEL = "llama32_3b"
GROUP = 128
SEED = 20260902
EVAL_STRIDE = 32
LANDSCAPE_TOKENS = 512
LANDSCAPE_CHANNELS = 2048


def import_project(repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(repo))
    from nar import activation_experiments as act
    from nar import extended_experiment as ext
    return act, ext


def select_site_layer(repo: Path) -> tuple[str, int, dict[str, float], list[dict[str, Any]]]:
    data = pd.read_csv(repo / "results" / MODEL / "e1c_per_layer.csv")
    selected = data[data.method.isin(["hadamard_full", "nar_kmax"])]
    pivot = selected.pivot(index=["site", "layer"], columns="method", values="mean_group_range").dropna()
    pivot["absolute_reduction"] = pivot.hadamard_full - pivot.nar_kmax
    pivot["relative_reduction"] = pivot.absolute_reduction / pivot.hadamard_full
    ranked = pivot.sort_values("absolute_reduction", ascending=False).reset_index()
    best = ranked.iloc[0]
    evidence = {
        "hadamard_mean_group_range": float(best.hadamard_full),
        "prismquant_mean_group_range": float(best.nar_kmax),
        "absolute_reduction": float(best.absolute_reduction),
        "relative_reduction_percent": 100.0 * float(best.relative_reduction),
    }
    top = [
        {
            "site": str(row.site),
            "layer": int(row.layer),
            "absolute_reduction": float(row.absolute_reduction),
            "relative_reduction_percent": 100.0 * float(row.relative_reduction),
        }
        for row in ranked.head(5).itertuples(index=False)
    ]
    return str(best.site), int(best.layer), evidence, top


def load_rows(ext: Any, mmap: np.memmap, indices: np.ndarray) -> torch.Tensor:
    bits = np.asarray(mmap[indices], dtype=np.uint16)
    return ext._bits_to_tensor(bits, torch.device("cpu")).float()


def select_hero(ext: Any, mmap: np.memmap, v1: torch.Tensor, seq_len: int) -> dict[str, Any]:
    positions = np.arange(EVAL_STRIDE, seq_len, EVAL_STRIDE, dtype=np.int64)
    projections: list[torch.Tensor] = []
    for sequence in range(mmap.shape[0]):
        bits = np.asarray(mmap[sequence, positions, :], dtype=np.uint16)
        rows = ext._bits_to_tensor(bits, torch.device("cpu")).float()
        projections.append(rows.mv(v1))
    projection = torch.cat(projections)
    absolute = projection.abs()
    q90 = float(torch.quantile(absolute, 0.90))
    q95 = float(torch.quantile(absolute, 0.95))
    row = int((absolute - q95).abs().argmin())
    sequence = row // len(positions)
    token = int(positions[row % len(positions)])
    if float(absolute[row]) < q90 or token == 0:
        raise AssertionError("hero selection is not a non-BOS top-decile v1 projection")
    return {
        "evaluation_rows": int(len(projection)),
        "selection_rule": "non-BOS stride-32 row nearest the 95th percentile of |projection on v1|",
        "top_decile_threshold_abs_projection": q90,
        "target_percentile_abs_projection": q95,
        "evaluation_row": row,
        "sequence_index": sequence,
        "token_position": token,
        "projection_on_v1": float(projection[row]),
        "absolute_projection_on_v1": float(absolute[row]),
    }


def centered_window(center: int, length: int, total: int, minimum: int = 0) -> tuple[int, int]:
    start = center - length // 2
    start = max(minimum, min(start, total - length))
    return int(start), int(start + length)


def group_aligned_channel_window(center_channel: int, n: int) -> tuple[int, int]:
    groups_wide = LANDSCAPE_CHANNELS // GROUP
    center_group = center_channel // GROUP
    start_group = max(0, min(n // GROUP - groups_wide, center_group - groups_wide // 2))
    return start_group * GROUP, (start_group + groups_wide) * GROUP


def transform_rows(
    act: Any,
    ext: Any,
    workdir: Path,
    site: str,
    layer: int,
    x: torch.Tensor,
    v1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, float, torch.Tensor]:
    n = x.shape[1]
    site_index = {"q_input": 0, "down_input": 1}[site]
    generator = torch.Generator(device="cpu").manual_seed(
        SEED + 1000 * layer + 10 * site_index + GROUP
    )
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64)
    signs = signs.float().mul_(2).sub_(1)
    hadamard = ext._full_hadamard_rows(x, signs)
    factor_name = "qkv" if site == "q_input" else "down"
    factor = act.RotationFactor.load(
        workdir / "activations" / MODEL / "activation_factors" / f"{factor_name}_layer_{layer:02d}.pt",
        torch.device("cpu"),
    )
    prism = factor.apply(x, signs)
    mapped_v1 = factor.apply(v1.reshape(1, -1), signs).reshape(-1, GROUP)
    group_energy = mapped_v1.square().sum(dim=-1)
    receiving_group = int(group_energy.argmax())
    captured = float(group_energy[receiving_group] / group_energy.sum())
    return hadamard, prism, receiving_group, captured, signs


def build_data(repo: Path, workdir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    act, ext = import_project(repo)
    site, layer, e1c, top_layers = select_site_layer(repo)
    if site != "down_input":
        raise AssertionError("the selected strongest E1c case is not the required down site")
    wide = workdir / "activations" / MODEL / "wide_cal_a"
    dump_meta = json.loads((wide / "DONE.json").read_text())
    mmap = ext._open_site(wide, dump_meta, site, layer)
    eig = torch.load(
        wide / "analysis" / "eigenspaces" / f"{site}_layer_{layer:02d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    v1 = eig["vectors"][:, 0].float()
    hero = select_hero(ext, mmap, v1, int(dump_meta["seq_len"]))
    token_start, token_stop = centered_window(
        int(hero["token_position"]), LANDSCAPE_TOKENS, int(dump_meta["seq_len"]), minimum=1
    )
    bits = np.asarray(
        mmap[int(hero["sequence_index"]), token_start:token_stop, :], dtype=np.uint16
    )
    raw = ext._bits_to_tensor(bits, torch.device("cpu")).float()
    hadamard, prism, receiving_group, capture, _signs = transform_rows(
        act, ext, workdir, site, layer, raw, v1
    )
    n = raw.shape[1]
    v1_peak_channel = int(v1.abs().argmax())
    raw_start, raw_stop = group_aligned_channel_window(v1_peak_channel, n)
    rotated_start = max(0, min(n - LANDSCAPE_CHANNELS, receiving_group * GROUP))
    rotated_start = (rotated_start // GROUP) * GROUP
    rotated_stop = rotated_start + LANDSCAPE_CHANNELS

    had_groups = hadamard.reshape(LANDSCAPE_TOKENS, -1, GROUP)
    prism_groups = prism.reshape(LANDSCAPE_TOKENS, -1, GROUP)
    had_range = had_groups.amax(dim=-1) - had_groups.amin(dim=-1)
    prism_range = prism_groups.amax(dim=-1) - prism_groups.amin(dim=-1)
    had_mean = float(had_range.mean())
    prism_mean = float(prism_range.mean())
    plotted_reduction = 100.0 * (had_mean - prism_mean) / had_mean
    p90_reduction = 100.0 * (
        float(torch.quantile(had_range, 0.90)) - float(torch.quantile(prism_range, 0.90))
    ) / float(torch.quantile(had_range, 0.90))
    if plotted_reduction < 10.0:
        raise AssertionError(
            f"PrismQuant range landscape is not visibly lower ({plotted_reduction:.1f}%); "
            "select the next strongest measured layer instead of altering data"
        )

    raw_group = v1_peak_channel // GROUP
    hero_offset = int(hero["token_position"]) - token_start
    traces = {
        "raw": raw[hero_offset, raw_group * GROUP : (raw_group + 1) * GROUP].numpy(),
        "hadamard": hadamard[
            hero_offset, receiving_group * GROUP : (receiving_group + 1) * GROUP
        ].numpy(),
        "nar_kmax": prism[
            hero_offset, receiving_group * GROUP : (receiving_group + 1) * GROUP
        ].numpy(),
    }
    token_axis = np.arange(token_start, token_stop, dtype=np.int32)
    arrays = {
        "raw_magnitude": raw[:, raw_start:raw_stop].abs().numpy(),
        "hadamard_magnitude": hadamard[:, rotated_start:rotated_stop].abs().numpy(),
        "hadamard_range": had_range.numpy(),
        "nar_kmax_range": prism_range.numpy(),
        "token_axis": token_axis,
        "trace_raw": traces["raw"],
        "trace_hadamard": traces["hadamard"],
        "trace_nar_kmax": traces["nar_kmax"],
    }
    metadata: dict[str, Any] = {
        "model": MODEL,
        "site": site,
        "layer": layer,
        "site_layer_selection_rule": "largest absolute measured mean-group-range reduction of nar_kmax versus hadamard_full across both E1c sites",
        "e1c_selected_case": e1c,
        "top_five_e1c_cases_by_absolute_reduction": top_layers,
        "group_size": GROUP,
        "groups": n // GROUP,
        "hero": hero,
        "token_window": {
            "sequence_index": int(hero["sequence_index"]),
            "position_start": token_start,
            "position_stop_exclusive": token_stop,
            "rows": LANDSCAPE_TOKENS,
            "bos_excluded": True,
        },
        "channel_windows": {
            "raw": [raw_start, raw_stop - 1],
            "hadamard_and_prismquant": [rotated_start, rotated_stop - 1],
            "width": LANDSCAPE_CHANNELS,
            "raw_window_rule": "group-aligned window centered on the largest-|loading| channel of frozen v1",
            "rotated_window_rule": "matched output-coordinate window beginning at the group receiving v1 under PrismQuant",
        },
        "group_window": [0, n // GROUP - 1],
        "v1_peak_channel": v1_peak_channel,
        "raw_trace_group": raw_group,
        "prismquant_receiving_group": receiving_group,
        "receiving_group_fraction_of_mapped_v1_energy": capture,
        "plotted_mean_ranges": {
            "hadamard": had_mean,
            "prismquant_kmax": prism_mean,
            "reduction_percent": plotted_reduction,
            "p90_reduction_percent": p90_reduction,
        },
        "trace_ranges": {
            name: float(values.max() - values.min()) for name, values in traces.items()
        },
        "trace_zero_points": {
            name: float(values.mean()) for name, values in traces.items()
        },
        "transform_seed": SEED,
        "font_family_resolved": resolved_serif_family(),
        "row1_series": LANDSCAPE_CHANNELS,
        "row2_series": n // GROUP,
        "row1_z_limits": {
            "raw": [0.0, float(arrays["raw_magnitude"].max())],
            "hadamard": [0.0, float(arrays["hadamard_magnitude"].max())],
        },
        "row2_shared_z_limits": [0.0, float(arrays["hadamard_range"].max())],
        "source": "frozen E1c dump, eigenspace, factor, and per-layer results; no model rerun",
    }
    return arrays, metadata


def style_3d(
    ax: plt.Axes,
    x_limits: tuple[int, int],
    token_limits: tuple[int, int],
    zmax: float,
    xlabel: str,
    zlabel: str,
) -> None:
    ax.set_xlim(*x_limits)
    ax.set_ylim(*token_limits)
    ax.set_zlim(0.0, zmax)
    ax.set_xticks(list(x_limits))
    ax.set_yticks(list(token_limits))
    ax.set_zticks([0.0, zmax])
    ax.set_xticklabels([str(x_limits[0]), f"{x_limits[1]}\u2003\u2003"])
    ax.set_yticklabels([f"\u2003\u2003{token_limits[0]}", str(token_limits[1])])
    ax.set_zticklabels(["0", f"{zmax:.2g}"])
    ax.set_xlabel(xlabel, fontsize=8.0, labelpad=-7)
    ax.set_ylabel("token", fontsize=8.0, labelpad=-8)
    ax.set_zlabel("")
    ax.tick_params(labelsize=7.0, pad=-2, length=1.8, width=0.45)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((2.4, 1.3, 0.9))
    try:
        ax.dist = 8.5
    except Exception:
        pass
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(PALETTE["pane_edge"])
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["axisline"]["color"] = PALETTE["text"]
        axis._axinfo["axisline"]["linewidth"] = 0.5


def render_landscape(
    values: np.ndarray,
    x_values: np.ndarray,
    token_values: np.ndarray,
    zmax: float,
    xlabel: str,
    zlabel: str,
    outbase: Path,
    mean_label: str | None = None,
) -> None:
    configure_style()
    fig = plt.figure(figsize=(3.2, 2.45))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0, 0, 1, 1])
    norm = Normalize(vmin=0.0, vmax=max(zmax, 1e-12))
    for column, x_value in enumerate(x_values):
        line = values[:, column]
        color = SEQUENTIAL_CMAP(norm(float(line.max())))
        ax.plot3D(
            np.full_like(token_values, x_value, dtype=np.float32),
            token_values,
            line,
            color=color,
            lw=0.7,
            alpha=0.95,
            solid_capstyle="butt",
            rasterized=values.shape[1] > 128,
        )
    style_3d(
        ax,
        (int(x_values[0]), int(x_values[-1])),
        (int(token_values[0]), int(token_values[-1])),
        zmax,
        xlabel,
        zlabel,
    )
    overlay = fig.add_axes([0, 0, 1, 1], zorder=100)
    overlay.patch.set_alpha(0.0)
    overlay.axis("off")
    overlay.text(0.965, 0.55, zlabel, fontsize=8.0, color=PALETTE["text"],
                 rotation=90, ha="center", va="center")
    if mean_label is not None:
        overlay.text(
            0.035, 0.955, mean_label, fontsize=7.5, color=PALETTE["text"],
            ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.4},
        )
    save_panel(fig, outbase)


def add_range_bracket(ax: plt.Axes, values: np.ndarray, color: str) -> None:
    low, high = float(values.min()), float(values.max())
    x = 134.0
    ax.vlines(x, low, high, color=color, lw=0.75)
    ax.hlines([low, high], 131.5, 136.5, color=color, lw=0.75)
    ax.text(
        139.0,
        0.5 * (low + high),
        f"{high - low:.3f}",
        color=PALETTE["text"],
        fontsize=7.5,
        ha="left",
        va="center",
    )


def render_trace(
    values: np.ndarray,
    method: str,
    y_limits: tuple[float, float],
    outbase: Path,
    ylabel: bool,
    xlabel: bool,
    zero_point: bool,
) -> None:
    configure_style()
    colors = {
        "raw": PALETTE["identity"],
        "hadamard": PALETTE["hadamard"],
        "nar_kmax": PALETTE["prismquant"],
    }
    fig, ax = plt.subplots(figsize=(1.8, 1.52))
    fig.subplots_adjust(left=0.28 if ylabel else 0.15, right=0.84, bottom=0.28, top=0.96)
    x = np.arange(GROUP)
    color = colors[method]
    ax.axhline(0.0, color=PALETTE["pane_edge"], lw=0.5, zorder=0)
    ax.plot(x, values, color=color, lw=1.0)
    if zero_point:
        mean = float(values.mean())
        ax.axhline(mean, color=PALETTE["reference"], lw=0.8, ls=(0, (3, 2)))
        pad = 0.035 * (y_limits[1] - y_limits[0])
        ax.text(3, mean + pad, "zero-point", fontsize=7.5, color=PALETTE["text"], ha="left", va="bottom")
    add_range_bracket(ax, values, color)
    ax.set_xlim(0, 154)
    ax.set_ylim(*y_limits)
    ax.set_xticks([0, 127])
    if ylabel:
        ax.set_ylabel("signed value", fontsize=8.0)
    else:
        ax.set_yticklabels([])
    if xlabel:
        ax.set_xlabel("channel in group", fontsize=8.0)
    clean_2d_axis(ax)
    save_panel(fig, outbase)


def write_summary_csvs(arrays: dict[str, np.ndarray], metadata: dict[str, Any], here: Path) -> None:
    trace_rows: list[dict[str, Any]] = []
    for method in ("raw", "hadamard", "nar_kmax"):
        values = arrays[f"trace_{method}"]
        for channel, value in enumerate(values):
            trace_rows.append(
                {
                    "method": method,
                    "channel_in_group": channel,
                    "signed_value": float(value),
                    "group_mean": float(values.mean()),
                    "group_range": float(values.max() - values.min()),
                }
            )
    pd.DataFrame(trace_rows).to_csv(here / "fig1_ranges.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for panel, method, values in (
        ("a", "raw", arrays["raw_magnitude"]),
        ("b", "hadamard", arrays["hadamard_magnitude"]),
        ("c", "hadamard", arrays["hadamard_range"]),
        ("d", "nar_kmax", arrays["nar_kmax_range"]),
    ):
        for line in range(values.shape[1]):
            summary_rows.append(
                {
                    "panel": panel,
                    "method": method,
                    "line_index": line,
                    "line_mean": float(values[:, line].mean()),
                    "line_max": float(values[:, line].max()),
                    "tokens": values.shape[0],
                }
            )
    pd.DataFrame(summary_rows).to_csv(here / "fig1_landscape_channels.csv", index=False)


def make_preview(here: Path) -> None:
    configure_style()
    fig = plt.figure(figsize=(6.6, 7.25))
    grid = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.72], hspace=0.02, wspace=0.02)
    placements = {
        "a": grid[0, 0:3], "b": grid[0, 3:6],
        "c": grid[1, 0:3], "d": grid[1, 3:6],
        "e": grid[2, 0:2], "f": grid[2, 2:4], "g": grid[2, 4:6],
    }
    for letter, slot in placements.items():
        ax = fig.add_subplot(slot)
        ax.imshow(plt.imread(here / f"fig1{letter}.png"))
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(here / "fig1_preview.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    arrays, metadata = build_data(args.repo.resolve(), args.workdir.resolve())
    token_axis = arrays["token_axis"]
    raw_start = int(metadata["channel_windows"]["raw"][0])
    rotated_start = int(metadata["channel_windows"]["hadamard_and_prismquant"][0])
    render_landscape(
        arrays["raw_magnitude"],
        np.arange(raw_start, raw_start + LANDSCAPE_CHANNELS),
        token_axis,
        float(arrays["raw_magnitude"].max()),
        "channel", "|x|", here / "fig1a",
    )
    render_landscape(
        arrays["hadamard_magnitude"],
        np.arange(rotated_start, rotated_start + LANDSCAPE_CHANNELS),
        token_axis,
        float(arrays["hadamard_magnitude"].max()),
        "channel", "|x|", here / "fig1b",
    )
    shared_range_z = float(arrays["hadamard_range"].max())
    render_landscape(
        arrays["hadamard_range"], np.arange(arrays["hadamard_range"].shape[1]), token_axis,
        shared_range_z, "group", "range", here / "fig1c",
        f"mean range {metadata['plotted_mean_ranges']['hadamard']:.2f}",
    )
    render_landscape(
        arrays["nar_kmax_range"], np.arange(arrays["nar_kmax_range"].shape[1]), token_axis,
        shared_range_z, "group", "range", here / "fig1d",
        f"mean range {metadata['plotted_mean_ranges']['prismquant_kmax']:.2f}",
    )
    trace_values = [arrays["trace_raw"], arrays["trace_hadamard"], arrays["trace_nar_kmax"]]
    trace_low = min(float(x.min()) for x in trace_values)
    trace_high = max(float(x.max()) for x in trace_values)
    margin = max(0.08 * (trace_high - trace_low), 1e-3)
    y_limits = (trace_low - margin, trace_high + margin)
    render_trace(arrays["trace_raw"], "raw", y_limits, here / "fig1e", True, False, False)
    render_trace(arrays["trace_hadamard"], "hadamard", y_limits, here / "fig1f", False, True, False)
    render_trace(arrays["trace_nar_kmax"], "nar_kmax", y_limits, here / "fig1g", False, False, True)
    write_summary_csvs(arrays, metadata, here)
    (here / "fig1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_preview(here)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
