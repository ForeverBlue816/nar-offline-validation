#!/usr/bin/env python3
"""Build Figure 1 from the frozen E1c activation dump.

The repository retains a full token-level E1c dump only for Llama-3.2-3B.
This script therefore labels that model explicitly and never substitutes it
silently for the requested 8B source.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors


RAW = "#7F7F7F"
HAD = "#E69F00"
NAR = "#0072B2"
GROUP = 128
SEED = 20260902


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def import_project(repo: Path) -> tuple[object, object]:
    sys.path.insert(0, str(repo))
    from nar import activation_experiments as act
    from nar import extended_experiment as ext

    return act, ext


def import_alignment_helper() -> object:
    helper = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(helper))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    return require_matplotlib_panel_alignment


def load_first_rows(wide: Path, layer: int, rows: int, ext: object) -> torch.Tensor:
    meta = json.loads((wide / "DONE.json").read_text())
    mmap = ext._open_site(wide, meta, "down_input", layer)
    flat = mmap.reshape(-1, int(meta["intermediate_size"]))[:rows]
    return ext._bits_to_tensor(flat, torch.device("cpu")).float()


def group_with_largest_abs(row: torch.Tensor) -> int:
    grouped = row.reshape(-1, GROUP)
    return int(grouped.abs().amax(dim=1).argmax())


def subtract_group_means(x: torch.Tensor) -> torch.Tensor:
    grouped = x.reshape(x.shape[0], -1, GROUP)
    return (grouped - grouped.mean(dim=-1, keepdim=True)).reshape_as(x)


def make_data(repo: Path, workdir: Path, csv_path: Path) -> dict[str, float | int | str]:
    act, ext = import_project(repo)
    model = "llama32_3b"
    wide = workdir / "activations" / model / "wide_cal_a"
    stats_path = workdir / "activations" / model / "e11_calibration" / "channel_stats.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)["activation_absmax"]
    layer_scores = []
    for key, values in stats.items():
        if key.startswith("down:"):
            layer_scores.append((float(values.max()), int(key.split(":")[1]), int(values.argmax())))
    global_max, layer, global_channel = max(layer_scores)

    x = load_first_rows(wide, layer, 64, ext)
    n = x.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1000 * layer + 10 + GROUP)
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1)
    h128 = ext.base.hadamard(GROUP, dtype=torch.float32)
    had = ext._block_hadamard_rows(x, GROUP, signs, h128)
    factor_path = workdir / "activations" / model / "activation_factors" / f"down_layer_{layer:02d}.pt"
    factor = act.RotationFactor.load(factor_path, torch.device("cpu"))
    nar = factor.apply(x, signs)
    nar_residual = subtract_group_means(nar)

    hero_token = int(x.abs().amax(dim=1).argmax())
    hero_raw_channel = int(x[hero_token].abs().argmax())
    method_values = {
        "identity": x,
        "hadamard_h128": had,
        "nar": nar,
        "nar_minus_group_mean": nar_residual,
    }
    group_index = {
        method: group_with_largest_abs(values[hero_token])
        for method, values in method_values.items()
    }
    group_index["nar_minus_group_mean"] = group_index["nar"]

    rows: list[dict[str, object]] = []
    for method, values in method_values.items():
        group = group_index[method]
        window = values[:, group * GROUP : (group + 1) * GROUP]
        means = window.mean(dim=-1)
        ranges = window.amax(dim=-1) - window.amin(dim=-1)
        for token in range(window.shape[0]):
            for channel in range(GROUP):
                rows.append(
                    {
                        "record_type": "activation",
                        "model": model,
                        "site": "down_input",
                        "layer": layer,
                        "method": method,
                        "token": token,
                        "group": group,
                        "relative_channel": channel,
                        "signed_value": float(window[token, channel]),
                        "group_mean": float(means[token]),
                        "group_range": float(ranges[token]),
                        "ecdf_probability": np.nan,
                        "source": "frozen E1c wide_cal_a",
                    }
                )

    ecdf_ranges: dict[str, torch.Tensor] = {}
    for method, values in (("hadamard_h128", had), ("nar_minus_group_mean", nar_residual)):
        ranges = values.reshape(64, -1, GROUP).amax(-1) - values.reshape(64, -1, GROUP).amin(-1)
        flat = torch.sort(ranges.flatten()).values
        ecdf_ranges[method] = flat
        probabilities = torch.arange(1, flat.numel() + 1, dtype=torch.float64) / flat.numel()
        for index, (value, probability) in enumerate(zip(flat, probabilities)):
            rows.append(
                {
                    "record_type": "ecdf",
                    "model": model,
                    "site": "down_input",
                    "layer": layer,
                    "method": method,
                    "token": index // (n // GROUP),
                    "group": index % (n // GROUP),
                    "relative_channel": np.nan,
                    "signed_value": np.nan,
                    "group_mean": np.nan,
                    "group_range": float(value),
                    "ecdf_probability": float(probability),
                    "source": "frozen E1c wide_cal_a",
                }
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    reduction = 1.0 - float(ecdf_ranges["nar_minus_group_mean"].mean() / ecdf_ranges["hadamard_h128"].mean())
    return {
        "model": model,
        "layer": layer,
        "global_absmax": global_max,
        "global_channel": global_channel,
        "hero_token": hero_token,
        "hero_raw_channel": hero_raw_channel,
        "groups": n // GROUP,
        "range_reduction": reduction,
        "anchor_error": factor.anchor_error,
    }


def panel_letter(ax: plt.Axes, letter: str) -> None:
    draw_text = ax.text2D if hasattr(ax, "text2D") else ax.text
    offset = mpl.transforms.ScaledTranslation(-11 / 72, 5 / 72, ax.figure.dpi_scale_trans)
    draw_text(0, 1, letter, transform=ax.transAxes + offset, fontsize=8.5, fontweight="bold", va="bottom")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_range_bracket(ax: plt.Axes, values: np.ndarray, color: str) -> None:
    lo, hi = float(values.min()), float(values.max())
    x = 124
    if hi - lo > 1e-6:
        ax.annotate("", xy=(x, hi), xytext=(x, lo), arrowprops={"arrowstyle": "<->", "color": color, "lw": 0.75})
    ax.text(0.68, 0.08, f"range = {hi-lo:.2f}", transform=ax.transAxes, color=color, fontsize=7.0, ha="right", va="bottom")


def render(csv_path: Path, outbase: Path, meta: dict[str, float | int | str]) -> None:
    configure_style()
    data = pd.read_csv(csv_path)
    activation = data[data.record_type.eq("activation")]
    fig = plt.figure(figsize=(5.5, 7.30))
    grid = fig.add_gridspec(4, 6, height_ratios=[1.10, 1.10, 1.38, 1.10], hspace=0.68, wspace=0.62)
    fig.subplots_adjust(left=0.11, right=0.90, bottom=0.10, top=0.97)
    hero_axes = [
        fig.add_subplot(grid[0, :3]),
        fig.add_subplot(grid[0, 3:]),
        fig.add_subplot(grid[1, :3]),
        fig.add_subplot(grid[1, 3:]),
    ]
    hero_methods = ["identity", "hadamard_h128", "nar", "nar_minus_group_mean"]
    hero_names = ["Identity: concentrated spike", "Random-sign H128: ± spread", "NAR: common-mode plateau", "NAR − group mean"]
    hero_colors = [RAW, HAD, NAR, NAR]
    all_hero = []
    for method in hero_methods:
        part = activation[(activation.method.eq(method)) & (activation.token.eq(meta["hero_token"]))]
        all_hero.extend(part.signed_value.to_numpy())
    limit = max(abs(np.percentile(all_hero, 0.2)), abs(np.percentile(all_hero, 99.8))) * 1.18
    limit = max(limit, max(abs(np.min(all_hero)), abs(np.max(all_hero))) * 1.05)
    for index, (ax, method, name, color) in enumerate(zip(hero_axes, hero_methods, hero_names, hero_colors)):
        part = activation[(activation.method.eq(method)) & (activation.token.eq(meta["hero_token"]))].sort_values("relative_channel")
        channel = part.relative_channel.to_numpy()
        value = part.signed_value.to_numpy()
        mean = float(part.group_mean.iloc[0])
        ax.axhline(0, color="#D0D0D0", lw=0.55, zorder=0)
        ax.plot(channel, value, color=color, lw=1.05 if method.startswith("nar") else 0.9)
        ax.fill_between(channel, 0, value, color=color, alpha=0.12)
        ax.axhline(mean, color=color, lw=0.85, ls=(0, (3, 2)))
        add_range_bracket(ax, value, color)
        ax.text(0.02, 0.90, name, transform=ax.transAxes, color=color, fontsize=7.5, fontweight="bold", va="top")
        ax.text(0.02, 0.76, f"zero point = mean = {mean:.2f}", transform=ax.transAxes, color=color, fontsize=7.0, va="top")
        ax.set_xlim(0, 127)
        ax.set_ylim(-limit, limit)
        ax.set_xticks([0, 32, 64, 96, 127])
        if index < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("channel within selected group")
        if index % 2 == 0:
            ax.set_ylabel("signed activation")
        else:
            ax.set_yticklabels([])
        clean_axis(ax)
        panel_letter(ax, chr(ord("a") + index))

    surface_axes = [fig.add_subplot(grid[2, 0:2], projection="3d"), fig.add_subplot(grid[2, 2:4], projection="3d"), fig.add_subplot(grid[2, 4:6], projection="3d")]
    surface_methods = ["identity", "hadamard_h128", "nar_minus_group_mean"]
    surface_names = ["Identity", "Random-sign H128", "NAR − group mean"]
    values_for_limit = []
    for method in surface_methods:
        part = activation[activation.method.eq(method)]
        values_for_limit.extend(part.signed_value.to_numpy())
    zlimit = max(abs(np.percentile(values_for_limit, 0.1)), abs(np.percentile(values_for_limit, 99.9)))
    zlimit = max(zlimit, max(abs(np.min(values_for_limit)), abs(np.max(values_for_limit))))
    non_raw = activation[activation.method.isin(["hadamard_h128", "nar_minus_group_mean"])].signed_value.abs().to_numpy()
    color_limit = max(float(np.percentile(non_raw, 99.7)), 1e-6)
    norm = colors.TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    channels = np.arange(GROUP)
    tokens = np.arange(64)
    xx, yy = np.meshgrid(channels, tokens)
    for index, (ax, method, name) in enumerate(zip(surface_axes, surface_methods, surface_names)):
        part = activation[activation.method.eq(method)].sort_values(["token", "relative_channel"])
        zz = part.signed_value.to_numpy().reshape(64, GROUP)
        ax.plot_surface(xx, yy, zz, cmap="RdBu_r", norm=norm, linewidth=0, antialiased=False, rstride=2, cstride=2, rasterized=True)
        ax.view_init(elev=25, azim=-60)
        ax.set_zlim(-zlimit, zlimit)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        panel_letter(ax, chr(ord("e") + index))

    fig.canvas.draw()
    for ax, name, color in zip(surface_axes, surface_names, [RAW, HAD, NAR]):
        bbox = ax.get_position()
        fig.text(bbox.x0, bbox.y1 + 0.006, name, color=color, fontsize=7.5, fontweight="bold", ha="left", va="bottom")
    surface_bottom = min(ax.get_position().y0 for ax in surface_axes)
    fig.text(0.50, surface_bottom - 0.010, f"channel 0–127 × token 0–63 · shared signed z: ±{zlimit:.0f} · color clipped: ±{color_limit:.1f}", color="#555555", fontsize=7.0, ha="center", va="top")

    ecdf_ax = fig.add_subplot(grid[3, :])
    label_probability = {"hadamard_h128": 0.62, "nar_minus_group_mean": 0.35}
    for method, color, name, lw in (("hadamard_h128", HAD, "Random-sign H128", 1.25), ("nar_minus_group_mean", NAR, "NAR − group mean", 1.8)):
        part = data[(data.record_type.eq("ecdf")) & (data.method.eq(method))].sort_values("group_range")
        ecdf_ax.plot(part.group_range, part.ecdf_probability, color=color, lw=lw)
        target = label_probability[method]
        point = part.iloc[int((part.ecdf_probability - target).abs().to_numpy().argmin())]
        ecdf_ax.annotate(name, xy=(float(point.group_range), float(point.ecdf_probability)), xytext=(7, 0), textcoords="offset points", color=color, fontsize=7.2, ha="left", va="center")
    if (data.loc[data.record_type.eq("ecdf"), "group_range"] <= 0).any():
        raise ValueError("ECDF group ranges must be positive for the logarithmic axis")
    ecdf_ax.set_xscale("log")
    ecdf_ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
    ecdf_ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    ecdf_ax.set_ylim(0, 1.02)
    ecdf_ax.set_xlabel("per-group range (max − min, log scale)")
    ecdf_ax.set_ylabel("empirical CDF")
    ecdf_ax.text(0.02, 0.95, f"mean range −{100*float(meta['range_reduction']):.1f}%", transform=ecdf_ax.transAxes, color=NAR, fontsize=7.5, fontweight="bold", va="top")
    ecdf_ax.text(0.02, 0.80, f"n = 64 × {meta['groups']} = {64*int(meta['groups']):,} paired groups", transform=ecdf_ax.transAxes, color="#333333", fontsize=7.0, va="top")
    clean_axis(ecdf_ax)
    panel_letter(ecdf_ax, "h")

    fig.text(0.995, 0.004, f"Llama-3.2-3B · down input · layer {meta['layer']} · frozen E1c rows", fontsize=7.0, color="#555555", ha="right", va="bottom")
    fig.canvas.draw()
    require_matplotlib_panel_alignment = import_alignment_helper()
    require_matplotlib_panel_alignment(
        fig,
        axes=[*hero_axes, *surface_axes, ecdf_ax],
        panel_ids=list("abcdefgh"),
        row_groups=[{"id": "hero-top", "panels": ["a", "b"]}, {"id": "hero-bottom", "panels": ["c", "d"]}, {"id": "surfaces", "panels": ["e", "f", "g"]}],
        require_panel_labels=True,
        json_out=outbase.with_suffix(".alignment.json"),
        overlay_svg=outbase.with_suffix(".alignment-overlay.svg"),
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
    csv_path = Path(__file__).with_name("fig1_ranges.csv")
    outbase = Path(__file__).with_name("fig1_pm_vs_plus")
    meta = make_data(args.repo.resolve(), args.workdir.resolve(), csv_path)
    Path(__file__).with_name("fig1_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    render(csv_path, outbase, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
