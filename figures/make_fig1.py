#!/usr/bin/env python3
"""Reproduce Figure 1 from frozen data and render its compact mechanism layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from figure_style import resolved_serif_family

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
    channel_medians = np.median(raw.abs().numpy(), axis=0)
    cumulative = np.r_[0, np.cumsum(channel_medians > 1.0)]
    window_counts = cumulative[LANDSCAPE_CHANNELS:] - cumulative[:-LANDSCAPE_CHANNELS]
    raw_start = int(np.argmax(window_counts))  # Earliest start wins a density tie.
    raw_stop = raw_start + LANDSCAPE_CHANNELS
    rotated_start, rotated_stop = raw_start, raw_stop

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
            "raw_window_rule": "maximize count of channels with median absolute activation > 1.0 over the plotted tokens; all stride-1 2048-wide windows; earliest start breaks ties",
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
    arrays["all_channel_median_magnitudes"] = channel_medians
    arrays["peak_density_window_counts"] = window_counts
    metadata["peak_density_selection"] = {
        "start_channel": raw_start, "stop_channel_exclusive": raw_stop,
        "qualifying_channel_count": int(window_counts[raw_start]),
        "median_threshold_strictly_greater_than": 1.0,
        "candidate_windows": int(len(window_counts)), "stride": 1,
        "tied_best_windows": int(np.sum(window_counts == window_counts.max())),
        "tie_break": "smallest start channel", "tokens": metadata["token_window"],
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
        if args.workdir is None:
            parser.error("--workdir is required without --reuse-data")
        arrays, metadata = build_data(args.repo.resolve(), args.workdir.resolve())

    hadamard_ranges = arrays["hadamard_range"]
    prismquant_ranges = arrays["nar_kmax_range"]
    metadata["plotted_mean_ranges"]["hadamard"] = float(hadamard_ranges.mean(dtype=np.float64))
    metadata["plotted_mean_ranges"]["prismquant_kmax"] = float(prismquant_ranges.mean(dtype=np.float64))
    metadata["plotted_mean_ranges"]["reduction_percent"] = 100 * (
        1 - metadata["plotted_mean_ranges"]["prismquant_kmax"]
        / metadata["plotted_mean_ranges"]["hadamard"]
    )
    metadata["range_statistics"] = {
        method: {
            "median": float(np.median(values)),
            "mean": float(values.mean(dtype=np.float64)),
            "percentile_95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
            "count": int(values.size),
        }
        for method, values in (
            ("hadamard", hadamard_ranges),
            ("prismquant_kmax", prismquant_ranges),
        )
    }
    metadata["trace_group_means"] = {
        name: float(arrays[f"trace_{name}"].mean())
        for name in ("raw", "hadamard", "nar_kmax")
    }
    metadata["trace_zero_points"] = {
        name: float(np.float16(arrays[f"trace_{name}"].min()))
        for name in ("raw", "hadamard", "nar_kmax")
    }
    metadata["zero_point_definition"] = (
        "Actual quantizer offset: fp16(min(values)), not the arithmetic group mean; "
        "dynamic_asym_int4 in nar/experiment.py."
    )
    metadata["row1_data_maxima"] = {
        "raw": float(arrays["raw_magnitude"].max()),
        "hadamard": float(arrays["hadamard_magnitude"].max()),
    }
    metadata["row1_z_limits"] = {"raw": [0.0, 40.0], "hadamard": [0.0, 4.0]}
    metadata.pop("row1_shared_z_limits", None)
    metadata["row2_shared_z_limits"] = [0.0, 10.0]
    metadata["correctness_resolution"]["fix"] = (
        "Color every segment by local height with a fixed row-wise normalization; "
        "put range summaries in metadata only."
    )
    metadata["correctness_resolution"]["old_z_clipping"] = (
        "PrismQuant maximum 8.9193306 exceeded the older 6.3898373 limit; "
        "the shared 0–10 scale covers both arrays."
    )
    if not args.reuse_data:
        np.savez_compressed(here / "fig1_source_arrays.npz", **arrays)

    trace_values = [arrays[f"trace_{name}"] for name in ("raw", "hadamard", "nar_kmax")]
    trace_low = min(float(v.min()) for v in trace_values)
    trace_high = max(float(v.max()) for v in trace_values)
    margin = max(.08 * (trace_high - trace_low), 1e-3)
    y_limits = (trace_low - margin, trace_high + margin)
    metadata.setdefault("trace_rendering", {})["y_limits"] = list(y_limits)
    metadata["rendered_panel_statistics"] = {
        letter: {"method": method, **metadata["range_statistics"][key], "shape": list(arrays[array_key].shape)}
        for letter, method, key, array_key in (
            ("c", "Hadamard", "hadamard", "hadamard_range"),
            ("d", "PrismQuant", "prismquant_kmax", "nar_kmax_range"))}
    write_summary_csvs(arrays, metadata, here)
    from fig1_layout import render_figure
    render_figure(arrays, metadata, here)
    (here / "fig1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["rendering_contract"], indent=2))


if __name__ == "__main__":
    main()
