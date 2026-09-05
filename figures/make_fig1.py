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
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
    pivot = selected.pivot(index=["site", "layer"], columns="method", values="mean_group_range")
    if not np.isfinite(pivot.to_numpy()).all():
        raise AssertionError("Incomplete or nonfinite site/layer candidates; no silent exclusions")
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
    rotated_start = raw_start
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
            "rotated_window_rule": "same numerical channel interval as raw; coordinates are in the rotated basis",
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
    for method, key in (("hadamard", "hadamard_range"), ("nar_kmax", "nar_kmax_range")):
        measured = float(np.ptp(arrays[f"trace_{method}"]))
        cell = float(arrays[key][hero_offset, receiving_group])
        if not np.isclose(measured, cell, rtol=1e-6):
            raise AssertionError("Trace range disagrees with its exact landscape cell")
    metadata["range_averaging"] = {
        "c_d": {"statistic": "arithmetic mean of every plotted max-minus-min value",
                "sequence": hero["sequence_index"], "tokens_inclusive": [token_start, token_stop - 1],
                "groups_inclusive": [0, n // GROUP - 1], "count": int(had_range.numel())},
        "e": {"statistic": "single raw token/group max-minus-min", "token": hero["token_position"], "group": raw_group},
        "f_g": {"statistic": "single rotated token/group max-minus-min", "token": hero["token_position"],
                "group": receiving_group, "landscape_row": hero_offset, "cell_identity_verified": True},
        "reconciliation": "c/d average 512 tokens × 64 groups; f/g select one cell (token 416, group 0) of those SAME arrays, so their values need not equal the means."
    }
    metadata["correctness_resolution"] = {
        "labels_or_arrays_swapped": False,
        "old_rendering_problem": "Whole polylines were colored by their maximum; opaque near lines obscure far lines. Mean range does not determine maximum or roughness.",
        "fix": "Color every segment by local height on shared c/d normalization; annotations computed inside renderer from its values array.",
    }
    metadata["row2_shared_z_limits"][1] = float(max(had_range.max(), prism_range.max()))
    # Centered covariance of exactly the 512 displayed tokens, computed via a
    # 512-by-512 Gram matrix; this is distinct from the uncentered E1c spectrum.
    centered = raw.double() - raw.double().mean(0)
    gram = centered @ centered.T / (len(raw) - 1)
    vals, u = torch.linalg.eigh(gram)
    eigvals = vals[-2:].flip(0)
    vectors = centered.T @ u[:, -2:].flip(1) / torch.sqrt((len(raw) - 1) * eigvals)
    scores = centered @ vectors
    mapped = act.RotationFactor.load(
        workdir / "activations" / MODEL / "activation_factors" / f"down_layer_{layer:02d}.pt",
        torch.device("cpu")).apply(vectors.T.float(), _signs)
    dc_cos = float(mapped[0, receiving_group * GROUP:(receiving_group + 1) * GROUP].sum() / np.sqrt(GROUP))
    metadata["geometry_covariance"] = {
        "definition": "centered sample covariance, divisor n-1; all 8192 channels of the same 512-token window",
        "top_two_eigenvalues": eigvals.tolist(), "rows": len(raw),
        "centered_pc1_cosine_with_prism_receiving_dc": abs(dc_cos),
        "frozen_second_moment_v1_dc_energy_fraction": capture,
        "raw_pc1_cosine_with_group_dc": abs(float(vectors[raw_group*GROUP:(raw_group+1)*GROUP,0].sum()/np.sqrt(GROUP))),
    }
    arrays["geometry_scores"] = scores.numpy()
    arrays["geometry_covariance_eigenvalues"] = eigvals.numpy()
    return arrays, metadata


def style_3d(ax, x_limits, token_limits, zmax, xlabel, zlabel):
    ax.set_xlim(*x_limits)
    ax.set_ylim(*token_limits)
    ax.set_zlim(0, zmax)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, integer=True, steps=[1, 2, 5, 10]))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel(xlabel, fontsize=7, labelpad=2, fontstyle="normal")
    ax.set_ylabel("token", fontsize=7, labelpad=0, fontstyle="normal")
    ax.set_zlabel("")
    ax.tick_params(labelsize=6, pad=-1, length=1.8, width=0.45)
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect((2.6, 1.2, 0.85), zoom=0.94)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((1, 1, 1, 1))
        axis.pane.set_edgecolor(PALETTE["pane_edge"])
        axis.pane.set_linewidth(0.6)
        axis._axinfo["grid"].update(color=PALETTE["grid"], linewidth=0.5)
        axis.line.set_color(PALETTE["pane_edge"])
        axis.line.set_linewidth(0.6)


def draw_landscape(ax, values, x_values, token_values, zmax, xlabel, mean=False):
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > zmax + 1e-6:
        raise AssertionError("Invalid or clipped landscape")
    # Every channel/group and every token is retained. Each adjacent segment
    # gets a local-height color, rather than one maximum-based color per line.
    coords = np.empty((values.shape[1], values.shape[0], 3), dtype=np.float32)
    coords[:, :, 0] = x_values[:, None]
    coords[:, :, 1] = token_values[None, :]
    coords[:, :, 2] = values.T
    segments = np.stack([coords[:, :-1], coords[:, 1:]], axis=2).reshape(-1, 2, 3)
    height = segments[:, :, 2].max(1)
    collection = Line3DCollection(segments, cmap=SEQUENTIAL_CMAP,
                                  norm=Normalize(0, zmax), linewidths=0.7)
    collection.set_array(height)
    # Dense marks are rasterized at 600 dpi; axes and labels stay vector.
    collection.set_rasterized(True)
    ax.add_collection3d(collection)
    style_3d(ax, (int(x_values[0]), int(x_values[-1])),
             (int(token_values[0]), int(token_values[-1])), zmax, xlabel, "")
    norm_text = f"{'shared' if mean else 'local'} height scale: 0–{zmax:.2f}"
    ax.text2D(0.04, 0.94, norm_text, transform=ax.transAxes, fontsize=7)
    if mean:
        value = float(values.mean(dtype=np.float64))
        ax.text2D(0.04, 0.85, f"mean range {value:.3f}", transform=ax.transAxes, fontsize=7)
        print(f"{ax.get_label()}: mean from plotted array = {value:.9f}", flush=True)


def render_landscape(values, x_values, token_values, zmax, xlabel, zlabel, outbase, mean_label=None):
    configure_style()
    fig = plt.figure(figsize=(3.2, 2.45))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", label=outbase.name)
    draw_landscape(ax, values, x_values, token_values, zmax, xlabel, mean_label is not None)
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
        fontsize=7.0,
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
    fig.subplots_adjust(left=0.28, right=0.84, bottom=0.28, top=0.96)
    x = np.arange(GROUP)
    color = colors[method]
    ax.hlines(0.0, 0, 127, color=PALETTE["pane_edge"], lw=0.5, zorder=0)
    ax.plot(x, values, color=color, lw=1.0)
    if zero_point:
        mean = float(np.float16(values.min()))
        ax.hlines(mean, 0, 127, color=PALETTE["reference"], lw=0.8, ls=(0, (3, 2)))
        pad = 0.035 * (y_limits[1] - y_limits[0])
        ax.text(3, mean - pad, "zero-point", fontsize=7.0, color=PALETTE["text"], ha="left", va="top")
    add_range_bracket(ax, values, color)
    ax.set_xlim(0, 154)
    ax.set_ylim(*y_limits)
    ax.set_xticks([0, 127])
    if ylabel:
        ax.set_ylabel("signed value", fontsize=7.0)
    else:
        ax.set_yticklabels([])
    if xlabel:
        ax.set_xlabel("channel in group", fontsize=7.0)
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
    """Assemble at 1:1 physical scale; preserve measured source plot areas."""
    from audit_panel_alignment import require_matplotlib_panel_alignment
    import pymupdf
    configure_style()
    width, height = 6.6, 7.3
    fig=plt.figure(figsize=(width,height))
    # Coordinates in inches, from the lower left. Every source panel keeps
    # its original dimensions: this never enlarges fonts by image rescaling.
    places={
        'a':(0,4.55,3.2,2.45), 'b':(3.4,4.55,3.2,2.45),
        'c':(0,1.95,3.2,2.45), 'd':(3.4,1.95,3.2,2.45),
        'e':(.1,.13,1.8,1.52), 'f':(2.4,.13,1.8,1.52), 'g':(4.7,.13,1.8,1.52),
    }
    titles={'a':'(a)  Raw · |x|','b':'(b)  Hadamard · |x|',
            'c':'(c)  Hadamard · group range','d':'(d)  PrismQuant · group range',
            'e':'(e)  Raw','f':'(f)  Hadamard','g':'(g)  PrismQuant'}
    plot_axes=[]
    for letter,(x,y,w,h) in places.items():
        ax=fig.add_axes([x/width,y/height,w/width,h/height])
        ax.imshow(plt.imread(here/f'fig1{letter}.png'),aspect='auto');ax.axis('off')
        source=json.loads((here/'qa'/f'fig1{letter}.alignment.json').read_text())
        x0,y0,x1,y1=source['layout']['panels'][0]['bbox_pt']
        measured=fig.add_axes([(x+x0/72)/width,(y+y0/72)/height,
                               (x1-x0)/72/width,(y1-y0)/72/height],label=letter)
        measured.axis('off');plot_axes.append(measured)
        label_y=y+h+.09
        fig.text((x+.12)/width,label_y/height,titles[letter],fontsize=7,va='top')
    require_matplotlib_panel_alignment(fig,json_out=here/'qa'/'fig1.alignment.json',
        axes=plot_axes,panel_ids=list('abcdefg'),row_groups=[['a','b'],['c','d'],['e','f','g']],
        column_groups=[['a','c'],['b','d']],strict=True)
    fig.savefig(here/'fig1_preview.png',dpi=300)
    plt.close(fig)
    # Page composition only: original matplotlib PDF content remains intact,
    # including editable text and vector axes. No raster replotting occurs.
    doc=pymupdf.open();page=doc.new_page(width=72*width,height=72*height)
    for letter,(x,y,w,h) in places.items():
        source=pymupdf.open(here/f'fig1{letter}.pdf')
        rect=pymupdf.Rect(x*72,(height-y-h)*72,(x+w)*72,(height-y)*72)
        page.show_pdf_page(rect,source,0)
        source.close()
    # Add reading-order labels using matplotlib, preserving the same font.
    labels=plt.figure(figsize=(width,height))
    for letter,(x,y,w,h) in places.items():
        label_y=y+h+.09
        labels.text((x+.12)/width,label_y/height,titles[letter],fontsize=7,va='top')
    import io
    buffer=io.BytesIO();labels.savefig(buffer,format='pdf',transparent=True)
    svg_buffer=io.StringIO();labels.savefig(svg_buffer,format='svg',transparent=True);plt.close(labels)
    source=pymupdf.open(stream=buffer.getvalue(),filetype='pdf');page.show_pdf_page(page.rect,source,0)
    doc.save(here/'fig1.pdf',deflate=True);doc.close();source.close()
    # SVG assembly uses the original matplotlib vectors and raster marks,
    # not a bitmap of the whole page. Nested SVG viewports preserve size.
    import xml.etree.ElementTree as ET
    ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
    root=ET.Element(f'{{{ns}}}svg',width=f'{72*width}pt',height=f'{72*height}pt',viewBox=f'0 0 {72*width} {72*height}')
    for letter,(x,y,w,h) in places.items():
        node=ET.parse(here/f'fig1{letter}.svg').getroot()
        # Prefix identifiers so gradients/clips cannot collide across panels.
        old_ids=[e.attrib['id'] for e in node.iter() if 'id' in e.attrib]
        raw=ET.tostring(node,encoding='unicode')
        for key in sorted(old_ids,key=len,reverse=True):
            raw=raw.replace(f'id="{key}"',f'id="{letter}_{key}"').replace(f'#{key})',f'#{letter}_{key})').replace(f'"#{key}"',f'"#{letter}_{key}"')
        node=ET.fromstring(raw);node.set('x',str(x*72));node.set('y',str((height-y-h)*72));root.append(node)
    root.append(ET.fromstring(svg_buffer.getvalue()))
    ET.ElementTree(root).write(here/'fig1.svg',encoding='unicode',xml_declaration=True)
    svg_path = here/'fig1.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    torch.set_num_threads(4)
    if args.reuse_data:
        arrays = dict(np.load(here / "fig1_source_arrays.npz"))
        metadata = json.loads((here / "fig1_metadata.json").read_text())
    else:
        if args.workdir is None: parser.error("--workdir is required without --reuse-data")
        arrays, metadata = build_data(args.repo.resolve(), args.workdir.resolve())
    metadata["plotted_mean_ranges"]["hadamard"] = float(arrays["hadamard_range"].mean(dtype=np.float64))
    metadata["plotted_mean_ranges"]["prismquant_kmax"] = float(arrays["nar_kmax_range"].mean(dtype=np.float64))
    metadata["plotted_mean_ranges"]["reduction_percent"] = 100 * (1 - metadata["plotted_mean_ranges"]["prismquant_kmax"] / metadata["plotted_mean_ranges"]["hadamard"])
    metadata["trace_group_means"] = {name: float(arrays[f"trace_{name}"].mean()) for name in ("raw", "hadamard", "nar_kmax")}
    metadata["trace_zero_points"] = {name: float(np.float16(arrays[f"trace_{name}"].min())) for name in ("raw", "hadamard", "nar_kmax")}
    metadata["zero_point_definition"] = "Actual quantizer offset: fp16(min(values)), not the arithmetic group mean; dynamic_asym_int4 in nar/experiment.py."
    metadata["correctness_resolution"]["old_z_clipping"] = "PrismQuant maximum 8.9193306 exceeded old common upper limit 6.3898373; common upper limit now covers both arrays."
    metadata["palette"] = PALETTE
    np.savez_compressed(here / "fig1_source_arrays.npz", **arrays)
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
    shared_range_z = float(metadata["row2_shared_z_limits"][1])
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
    metadata["rendered_panel_statistics"] = {letter: {"method": method, "mean_from_plotted_array": float(arrays[key].mean(dtype=np.float64)), "max": float(arrays[key].max()), "shape": list(arrays[key].shape)} for letter, method, key in [("c", "Hadamard", "hadamard_range"), ("d", "PrismQuant", "nar_kmax_range")]}
    write_summary_csvs(arrays, metadata, here)
    (here / "fig1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_preview(here)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
