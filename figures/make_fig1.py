#!/usr/bin/env python3
"""Render Figure 1 from the frozen Llama-3.2-3B E1c down-input dump."""

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
import torch
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from palette import get_palette, resolved_serif_family
GROUP = 128
LAYER = 1
TOKEN_ROWS = 1024
TOKEN_STRIDE = 32
COLOR_CHUNK_ROWS = 128
SEED = 20260902


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
            "legend.fontsize": 8.0,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def import_project(repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(repo))
    from nar import activation_experiments as act
    from nar import extended_experiment as ext

    return act, ext


def import_alignment_helper() -> Any:
    helper = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(helper))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def sampled_non_bos_rows(wide: Path, ext: Any) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict[str, Any]]:
    meta = json.loads((wide / "DONE.json").read_text())
    mmap = ext._open_site(wide, meta, "down_input", LAYER)
    positions = np.arange(TOKEN_STRIDE, int(meta["seq_len"]), TOKEN_STRIDE, dtype=np.int64)
    sequence_index = np.repeat(np.arange(int(meta["sequences"]), dtype=np.int64), positions.size)
    token_position = np.tile(positions, int(meta["sequences"]))
    selected = mmap[:, positions, :].reshape(-1, int(meta["intermediate_size"]))[:TOKEN_ROWS]
    x = ext._bits_to_tensor(selected, torch.device("cpu")).float()
    if x.shape != (TOKEN_ROWS, 8192):
        raise AssertionError(f"unexpected Figure 1 tensor shape: {tuple(x.shape)}")
    if np.any(token_position[:TOKEN_ROWS] == 0):
        raise AssertionError("BOS position 0 leaked into Figure 1")
    return x, sequence_index[:TOKEN_ROWS], token_position[:TOKEN_ROWS], meta


def subtract_group_means(x: torch.Tensor) -> torch.Tensor:
    grouped = x.reshape(x.shape[0], -1, GROUP)
    return (grouped - grouped.mean(dim=-1, keepdim=True)).reshape_as(x)


def receiving_group(factor: Any, signs: torch.Tensor, channel: int) -> tuple[int, float]:
    basis = torch.zeros((1, factor.n), dtype=torch.float32)
    basis[0, channel] = 1.0
    mapped = factor.apply(basis, signs).reshape(-1, GROUP)
    energy = mapped.square().sum(-1)
    group = int(energy.argmax())
    return group, float(energy[group])


def signed_peak(values: np.ndarray, axis: int) -> np.ndarray:
    maxima = np.max(values, axis=axis)
    minima = np.min(values, axis=axis)
    return np.where(np.abs(maxima) >= np.abs(minima), maxima, minima)


def make_data(repo: Path, workdir: Path, trace_csv: Path, landscape_csv: Path) -> dict[str, Any]:
    act, ext = import_project(repo)
    model = "llama32_3b"
    wide = workdir / "activations" / model / "wide_cal_a"
    x, sequences, positions, dump_meta = sampled_non_bos_rows(wide, ext)
    n = x.shape[1]

    persistent = x.abs().median(dim=0).values
    strong_channel = int(persistent.argmax())
    hero_row = int(x[:, strong_channel].abs().argmax())

    generator = torch.Generator(device="cpu").manual_seed(
        SEED + 1000 * LAYER + 10 + GROUP
    )
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64)
    signs = signs.float().mul_(2).sub_(1)
    h128 = ext.base.hadamard(GROUP, dtype=torch.float32)
    had = ext._block_hadamard_rows(x, GROUP, signs, h128)

    factor_path = workdir / "activations" / model / "activation_factors" / f"down_layer_{LAYER:02d}.pt"
    factor = act.RotationFactor.load(factor_path, torch.device("cpu"))
    nar = factor.apply(x, signs)
    nar_residual = subtract_group_means(nar)
    target_group, target_group_energy = receiving_group(factor, signs, strong_channel)

    transforms = {
        "raw": x,
        "random_rotation": had,
        "nar_kmax": nar,
        "nar_kmax_zero_point_removed": nar_residual,
    }
    source_group = strong_channel // GROUP
    trace_groups = {
        "raw": source_group,
        "random_rotation": source_group,
        "nar_kmax": target_group,
        "nar_kmax_zero_point_removed": target_group,
    }

    trace_rows: list[dict[str, Any]] = []
    for method, tensor in transforms.items():
        group = trace_groups[method]
        values = tensor[hero_row, group * GROUP : (group + 1) * GROUP]
        mean = float(values.mean())
        value_range = float(values.max() - values.min())
        for relative_channel, value in enumerate(values.tolist()):
            trace_rows.append(
                {
                    "model": model,
                    "site": "down_input",
                    "layer": LAYER,
                    "method": method,
                    "sample_row": hero_row,
                    "sequence_index": int(sequences[hero_row]),
                    "token_position": int(positions[hero_row]),
                    "source_persistent_channel": strong_channel,
                    "display_group": group,
                    "relative_channel": relative_channel,
                    "signed_value": float(value),
                    "group_mean": mean,
                    "group_range": value_range,
                    "group_size": GROUP,
                    "bos_excluded": True,
                }
            )
    pd.DataFrame(trace_rows).to_csv(trace_csv, index=False)

    landscape_frames: list[pd.DataFrame] = []
    chunks = TOKEN_ROWS // COLOR_CHUNK_ROWS
    if TOKEN_ROWS % COLOR_CHUNK_ROWS:
        raise AssertionError("token rows must divide evenly into landscape color chunks")
    for method, tensor in transforms.items():
        values = tensor.numpy()
        channel_median = np.median(values, axis=0)
        channel_abs_median = np.median(np.abs(values), axis=0)
        panel_absmax = float(np.abs(values).max())
        for chunk in range(chunks):
            start = chunk * COLOR_CHUNK_ROWS
            stop = (chunk + 1) * COLOR_CHUNK_ROWS
            peaks = signed_peak(values[start:stop], axis=0)
            landscape_frames.append(
                pd.DataFrame(
                    {
                        "model": np.repeat(model, n),
                        "site": np.repeat("down_input", n),
                        "layer": np.repeat(LAYER, n),
                        "method": np.repeat(method, n),
                        "channel": np.arange(n, dtype=np.int32),
                        "token_row_start": np.repeat(start, n),
                        "token_row_stop_exclusive": np.repeat(stop, n),
                        "signed_color_value": peaks,
                        "channel_signed_median": channel_median,
                        "channel_median_abs": channel_abs_median,
                        "panel_absmax": np.repeat(panel_absmax, n),
                        "bos_excluded": np.repeat(True, n),
                    }
                )
            )
    pd.concat(landscape_frames, ignore_index=True).to_csv(landscape_csv, index=False)

    nar_groups = nar.reshape(TOKEN_ROWS, -1, GROUP)
    plateau_means = nar_groups.mean(-1)
    bright_group = int(plateau_means.abs().amax(dim=0).argmax())
    shared_absmax = float(had.abs().max())
    plateau_peak = float(plateau_means[:, bright_group].abs().max())
    plateau_visible = plateau_peak / max(shared_absmax, 1e-30) >= 0.08

    metadata: dict[str, Any] = {
        "model": model,
        "site": "down_input",
        "layer": LAYER,
        "channels": n,
        "group_size": GROUP,
        "token_stride": TOKEN_STRIDE,
        "source_rows_before_bos_exclusion": int(dump_meta["sequences"] * (dump_meta["seq_len"] // TOKEN_STRIDE)),
        "rows_after_bos_exclusion": int(dump_meta["sequences"] * (dump_meta["seq_len"] // TOKEN_STRIDE - 1)),
        "plotted_token_rows": TOKEN_ROWS,
        "bos_exclusion_rule": "exclude sequence position 0 from every sequence before deterministic row selection",
        "strongest_persistent_channel": strong_channel,
        "strongest_channel_median_abs": float(persistent[strong_channel]),
        "hero_sample_row": hero_row,
        "hero_sequence_index": int(sequences[hero_row]),
        "hero_token_position": int(positions[hero_row]),
        "hero_raw_value": float(x[hero_row, strong_channel]),
        "source_group": source_group,
        "nar_receiving_group": target_group,
        "nar_receiving_group_basis_energy_fraction": target_group_energy,
        "brightest_nar_plateau_group": bright_group,
        "brightest_nar_plateau_peak_abs_mean": plateau_peak,
        "shared_rotated_z_absmax": shared_absmax,
        "nar_plateau_visible_ratio": plateau_peak / max(shared_absmax, 1e-30),
        "zoom_strip_added": not plateau_visible,
        "panel_absmax": {method: float(tensor.abs().max()) for method, tensor in transforms.items()},
        "nar_anchor_error": float(factor.anchor_error),
        "trace_groups": trace_groups,
        "transform_seed": SEED,
        "resolved_font_family": resolved_serif_family(),
        "landscape_color_rule": (
            "coolwarm; symmetric per-panel normalization to that panel's full absmax; "
            "each channel line is locally colored by the signed maximum-absolute value "
            f"within consecutive {COLOR_CHUNK_ROWS}-row segments; no color clipping"
        ),
        "source": "frozen E1c wide calibration dump and frozen k=max factor",
    }
    return metadata


def load_landscape_tensors(repo: Path, workdir: Path, meta: dict[str, Any]) -> dict[str, torch.Tensor]:
    act, ext = import_project(repo)
    wide = workdir / "activations" / "llama32_3b" / "wide_cal_a"
    x, _sequences, _positions, _dump_meta = sampled_non_bos_rows(wide, ext)
    n = x.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1000 * LAYER + 10 + GROUP)
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64)
    signs = signs.float().mul_(2).sub_(1)
    h128 = ext.base.hadamard(GROUP, dtype=torch.float32)
    had = ext._block_hadamard_rows(x, GROUP, signs, h128)
    factor = act.RotationFactor.load(
        workdir / "activations" / "llama32_3b" / "activation_factors" / f"down_layer_{LAYER:02d}.pt",
        torch.device("cpu"),
    )
    nar = factor.apply(x, signs)
    return {
        "raw": x,
        "random_rotation": had,
        "nar_kmax": nar,
        "nar_kmax_zero_point_removed": subtract_group_means(nar),
    }


def panel_letter(ax: plt.Axes, letter: str) -> None:
    draw_text = ax.text2D if hasattr(ax, "text2D") else ax.text
    offset = mpl.transforms.ScaledTranslation(-8 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    draw_text(
        0,
        1,
        letter,
        transform=ax.transAxes + offset,
        fontsize=9.0,
        fontweight="bold",
        color="#000000",
        va="bottom",
        ha="right",
    )


def style_3d(ax: plt.Axes, zlim: float, first: bool) -> None:
    ax.set_xlim(0, 8191)
    ax.set_ylim(0, TOKEN_ROWS - 1)
    ax.set_zlim(-zlim, zlim)
    ax.set_xticks([0, 8191])
    ax.set_yticks([0, TOKEN_ROWS - 1])
    ax.set_zticks([-zlim, zlim])
    ax.set_xticklabels(["0", "8191"])
    ax.set_yticklabels(["0", "1023"])
    ax.set_zticklabels([f"−{zlim:.2g}", f"{zlim:.2g}"])
    ax.tick_params(pad=-2, width=0.4, length=1.8)
    ax.set_xlabel("channel", labelpad=-8)
    ax.set_ylabel("token", labelpad=-9)
    if first:
        ax.set_zlabel("value", labelpad=-8)
    else:
        ax.set_zlabel("")
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1.25, 1.0, 0.62))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.94, 0.94, 0.94, 1.0))
        axis.pane.set_edgecolor((0.72, 0.72, 0.72, 1.0))
        axis._axinfo["grid"]["linewidth"] = 0.30
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.6)
        axis._axinfo["axisline"]["linewidth"] = 0.45
        axis.label.set_rasterized(True)
        for tick_label in axis.get_ticklabels():
            tick_label.set_rasterized(True)


def add_landscape(ax: plt.Axes, values: np.ndarray, zlim: float, first: bool) -> None:
    token_count, channels = values.shape
    chunk = COLOR_CHUNK_ROWS
    chunk_count = token_count // chunk
    point_count = chunk + 1
    segments = np.empty((channels, chunk_count, point_count, 3), dtype=np.float32)
    channel_axis = np.arange(channels, dtype=np.float32)
    color_values = np.empty((channels, chunk_count), dtype=np.float32)
    for part in range(chunk_count):
        start = part * chunk
        stop = (part + 1) * chunk
        source_stop = min(stop + 1, token_count)
        local = values[start:source_stop]
        if local.shape[0] < point_count:
            local = np.concatenate((local, local[-1:]), axis=0)
        segments[:, part, :, 0] = channel_axis[:, None]
        token_axis = np.arange(start, start + point_count, dtype=np.float32)
        token_axis[-1] = min(float(token_axis[-1]), float(token_count - 1))
        segments[:, part, :, 1] = token_axis[None, :]
        segments[:, part, :, 2] = local.T
        color_values[:, part] = signed_peak(values[start:stop], axis=0)
    absmax = max(float(np.abs(values).max()), 1e-12)
    norm = colors.TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    collection = Line3DCollection(
        segments.reshape(-1, point_count, 3),
        cmap="coolwarm",
        norm=norm,
        linewidths=0.4,
        alpha=0.9,
        rasterized=True,
    )
    collection.set_array(color_values.reshape(-1))
    ax.add_collection3d(collection)
    style_3d(ax, zlim, first)


def clean_2d(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_range_bracket(ax: plt.Axes, values: np.ndarray, color: str) -> None:
    lo, hi = float(values.min()), float(values.max())
    x = 124.0
    cap = 2.2
    ax.vlines(x, lo, hi, color=color, lw=0.75)
    ax.hlines([lo, hi], x - cap, x + cap, color=color, lw=0.75)



def render(
    repo: Path,
    workdir: Path,
    trace_csv: Path,
    outbase: Path,
    metadata: dict[str, Any],
    palette_name: str,
) -> None:
    configure_style()
    palette = get_palette(palette_name)
    raw_color = palette["identity"]
    had_color = palette["hadamard"]
    prism_color = palette["prismquant"]
    tensors = load_landscape_tensors(repo, workdir, metadata)
    trace = pd.read_csv(trace_csv)
    methods = ["raw", "random_rotation", "nar_kmax", "nar_kmax_zero_point_removed"]
    titles = ["raw", "random rotation", "PrismQuant (ours)", "PrismQuant\nzero-point removed"]
    method_colors = [raw_color, had_color, prism_color, prism_color]

    fig = plt.figure(figsize=(5.50, 4.20))
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.78, 1.0],
        left=0.105,
        right=0.935,
        bottom=0.175,
        top=0.925,
        hspace=0.52,
        wspace=0.42,
    )
    landscape_axes = [fig.add_subplot(grid[0, index], projection="3d") for index in range(4)]
    trace_axes = [fig.add_subplot(grid[1, index]) for index in range(4)]

    shared_z = float(metadata["shared_rotated_z_absmax"])
    for index, (ax, method, title) in enumerate(zip(landscape_axes, methods, titles)):
        values = tensors[method].numpy()
        zlim = float(np.abs(values).max()) if method == "raw" else shared_z
        add_landscape(ax, values, zlim, first=index == 0)
        ax.set_title(title, fontsize=8.0, pad=1.5, color=method_colors[index], linespacing=1.2)
        panel_letter(ax, chr(ord("a") + index))

    if bool(metadata["zoom_strip_added"]):
        parent = landscape_axes[2]
        group = int(metadata["brightest_nar_plateau_group"])
        center = group * GROUP + GROUP // 2
        start = max(0, min(8192 - 512, center - 256))
        inset = parent.inset_axes([0.08, -0.06, 0.84, 0.19])
        zoom = tensors["nar_kmax"][:, start : start + 512].numpy()
        mean = zoom.reshape(TOKEN_ROWS, 4, GROUP).mean(-1)
        inset.plot(np.arange(4), np.median(mean, axis=0), color=prism_color, lw=1.0, marker="o", ms=2.2)
        inset.axhline(0, color="#A0A0A0", lw=0.45)
        inset.set_xticks([0, 3], [str(start), str(start + 511)])
        inset.set_yticks([])
        inset.set_title("512-channel plateau zoom", fontsize=7.5, pad=0.5, color=prism_color)
        clean_2d(inset)

    for index, (ax, method, color) in enumerate(zip(trace_axes, methods, method_colors)):
        part = trace[trace.method.eq(method)].sort_values("relative_channel")
        x = part.relative_channel.to_numpy()
        y = part.signed_value.to_numpy()
        mean = float(part.group_mean.iloc[0])
        ax.axhline(0, color="#C8C8C8", lw=0.55, zorder=0)
        ax.plot(x, y, color=color, lw=1.2)
        ax.axhline(mean, color=color, lw=0.8, ls=(0, (3, 2)))
        ax.text(0.50, 0.94, "zero-point", transform=ax.transAxes,
                color="#000000", fontsize=7.5, ha="center", va="top")
        add_range_bracket(ax, y, color)
        span = float(y.max()) - float(y.min())
        lower_margin = max(0.08 * span, 0.004)
        upper_margin = max(0.34 * span, 0.012)
        ax.set_ylim(float(y.min()) - lower_margin, float(y.max()) + upper_margin)
        ax.set_xlim(0, 127)
        ax.set_xticks([0, 127])
        ax.set_xlabel("")
        if index == 0:
            ax.set_ylabel("signed value")
        else:
            ax.set_yticklabels([])
        group = int(part.display_group.iloc[0])
        value_range = float(y.max() - y.min())
        ax.set_title(f"group {group} · range {value_range:.3f}", fontsize=7.5,
                     color="#000000", loc="center", pad=2.0)
        clean_2d(ax)
        panel_letter(ax, chr(ord("e") + index))

    fig.text(
        0.105,
        0.032,
        (
            f"Llama-3.2-3B · down input · layer {LAYER} · seq {metadata['hero_sequence_index']}, "
            f"token {metadata['hero_token_position']} · channel {metadata['strongest_persistent_channel']}\n"
            f"1,024 rows × 8,192 channels · BOS excluded"
        ),
        fontsize=7.0,
        color="#000000",
        ha="left",
        va="bottom",
    )
    fig.text(0.52, 0.130, "channel within group", fontsize=8.0, color="#000000", ha="center", va="top")

    fig.canvas.draw()
    require_matplotlib_panel_alignment = import_alignment_helper()
    require_matplotlib_panel_alignment(
        fig,
        axes=[*landscape_axes, *trace_axes],
        panel_ids=list("abcdefgh"),
        row_groups=[
            {"id": "landscapes", "panels": list("abcd")},
            {"id": "traces", "panels": list("efgh")},
        ],
        column_groups=[
            {"id": f"column-{i}", "panels": [top, bottom]}
            for i, (top, bottom) in enumerate(zip("abcd", "efgh"))
        ],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    trace_csv = here / "fig1_ranges.csv"
    landscape_csv = here / "fig1_landscape_channels.csv"
    metadata = make_data(args.repo.resolve(), args.workdir.resolve(), trace_csv, landscape_csv)
    for palette_name in ("A", "B"):
        render(args.repo.resolve(), args.workdir.resolve(), trace_csv,
               here / f"fig1_variant{palette_name}", metadata, palette_name)
    for suffix in (".svg", ".pdf", ".png", ".alignment.json", ".alignment-overlay.svg"):
        shutil.copyfile(here / f"fig1_variantA{suffix}", here / f"fig1_pm_vs_plus{suffix}")
    (here / "fig1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    caption = (
        "Persistent activation outliers become removable common mode. "
        "Signed down-projection inputs are shown for 1,024 stride-32 token rows and all 8,192 channels "
        "before rotation, after random-sign block-H128, after PrismQuant k=max alignment, and after subtracting each "
        "128-channel group's mean. BOS position 0 is excluded from every sequence. The lower row shows "
        f"the paired non-BOS token at sequence {metadata['hero_sequence_index']}, position "
        f"{metadata['hero_token_position']}, selected on persistent channel "
        f"{metadata['strongest_persistent_channel']}; dashed lines are group zero points and brackets "
        "report max−min range. Colors are normalized independently to each panel's own absolute maximum; "
        "panels b–d share b's signed z limits."
    )
    (here / "fig1_caption.txt").write_text(caption + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
