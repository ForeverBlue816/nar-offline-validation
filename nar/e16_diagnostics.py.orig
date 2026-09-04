#!/usr/bin/env python3
"""E16 offline SmoothQuant error and DC-alignment diagnostics.

Uses the frozen E11 calibration chunks/statistics and one seed.  It does not
evaluate or extract official DuQuant artifacts, following the later
citation-only protocol amendment; the already implemented E11 DuQuant-style
construction is retained as the matched diagnostic baseline.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Callable

import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from . import e11_fair_baselines as e11
except ImportError:
    import activation_experiments as act
    import experiment as base
    import e11_fair_baselines as e11


METHODS = ("hadamard", "smoothquant_hadamard", "duquant_style", "nar")


def _factor_path(workdir: Path, model_key: str, site: str, layer: int) -> Path:
    return act.factor_dir(workdir, model_key) / f"{site}_layer_{layer:02d}.pt"


def reconstruct_directions(factor: act.RotationFactor, count: int) -> torch.Tensor:
    """Recover original eigendirection rows from anchors using G(V)^T."""
    rows = torch.zeros((count, factor.n), device=factor.reflectors.device, dtype=torch.float32)
    rows[torch.arange(count, device=rows.device), torch.arange(count, device=rows.device) * factor.b] = 1.0
    active_indices = [index for index in range(factor.reflectors.shape[0]) if bool(factor.active[index])]
    for index in reversed(active_indices):
        vector = factor.reflectors[index]
        rows = rows - 2 * (rows @ vector).unsqueeze(-1) * vector
    return rows


def dc_fraction(transformed: torch.Tensor, denominator: torch.Tensor, group: int = 128) -> torch.Tensor:
    blocks = transformed.float().reshape(transformed.shape[0], -1, group)
    dc_energy = (blocks.sum(-1).square() / group).sum(-1)
    return dc_energy / denominator.float().square().sum(-1).clamp_min(1e-30)


class OfflineCollector:
    def __init__(self, model: torch.nn.Module, had: e11.Transform, smooth: e11.Transform, layers: int):
        self.model = model
        self.transforms = {"hadamard": had, "smoothquant_hadamard": smooth}
        self.layers = layers
        self.handles: list[Any] = []
        self.values: dict[tuple[str, str, int, str], torch.Tensor] = {}

    def consume(self, site: str, layer: int, value: torch.Tensor) -> None:
        for method, transform in self.transforms.items():
            rotated = transform.activation(site, layer, value).float()
            groups = base.group_view(rotated, 128)
            dequant, _, _, _ = base.dynamic_asym_int4(rotated, 128)
            entries = {
                "range_sum": (groups.amax(-1) - groups.amin(-1)).double().sum(),
                "groups": torch.tensor(groups.numel() // 128, device=rotated.device, dtype=torch.float64),
                "error": (dequant.float() - rotated).square().double().sum(),
                "energy": rotated.square().double().sum(),
            }
            for name, number in entries.items():
                key = (method, site, layer, name)
                if key not in self.values:
                    self.values[key] = torch.zeros((), device=rotated.device, dtype=torch.float64)
                self.values[key] += number

    def q_hook(self, layer: int) -> Callable[..., torch.Tensor]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            self.consume("qkv", layer, output)
            return output
        return hook

    def down_hook(self, layer: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            self.consume("down", layer, inputs[0])
        return hook

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers[:self.layers]):
            self.handles.append(block.input_layernorm.register_forward_hook(self.q_hook(layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.down_hook(layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def rows(self, model_key: str) -> list[dict[str, Any]]:
        result = []
        for method in self.transforms:
            for site in ("qkv", "down"):
                for layer in range(self.layers):
                    get = lambda name: float(self.values[(method, site, layer, name)].cpu())
                    result.append({
                        "model": model_key, "method": method, "site": site, "layer": layer,
                        "mean_group_range": get("range_sum") / get("groups"),
                        "nmse": get("error") / get("energy"),
                        "groups_observed": int(get("groups")),
                    })
        return result


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E16 diagnostics require CUDA")
    workdir = Path(args.workdir).resolve()
    model_id, model_key = act.model_id_and_key(args.model)
    base.setup_logging(workdir, f"e16-diagnostics-{model_key}")
    result_dir = workdir / "results" / model_key
    done = result_dir / "E16_DIAGNOSTICS_DONE.json"
    if done.exists():
        return
    calibration = e11.calibration_dir(workdir, model_key)
    meta = json.loads((calibration / "DONE.json").read_text())
    stats = torch.load(calibration / "channel_stats.pt", map_location="cpu", weights_only=True)
    model = base.load_model(model_id, workdir)
    layers = int(model.config.num_hidden_layers)
    dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
    device = torch.device("cuda")
    had = e11.Transform("hadamard_g128_asym", model_key, workdir, 0, args.seed, layers, dimensions, device, stats)
    smooth = e11.Transform("smoothquant_hadamard_g128_asym", model_key, workdir, 0, args.seed, layers, dimensions, device, stats)
    duquant = e11.Transform("duquant_style_g128_asym", model_key, workdir, 0, args.seed, layers, dimensions, device, stats)

    alignment_rows: list[dict[str, Any]] = []
    for layer in range(layers):
        factor = act.RotationFactor.load(_factor_path(workdir, model_key, "qkv", layer), device)
        directions = reconstruct_directions(factor, 8)
        top_channel = int(stats["activation_absmax"][e11._key("qkv", layer)].argmax())
        channel = torch.zeros((1, dimensions["qkv"]), device=device)
        channel[0, top_channel] = 1.0
        for method in METHODS:
            if method == "hadamard":
                transformed = had.activation("qkv", layer, directions)
                transformed_channel = had.activation("qkv", layer, channel)
            elif method == "smoothquant_hadamard":
                transformed = smooth.activation("qkv", layer, directions)
                transformed_channel = smooth.activation("qkv", layer, channel)
            elif method == "duquant_style":
                transformed = duquant.activation("qkv", layer, directions)
                transformed_channel = duquant.activation("qkv", layer, channel)
            else:
                transformed = factor.apply(directions, had.signs[("qkv", layer)])
                transformed_channel = factor.apply(channel, had.signs[("qkv", layer)])
            scores = dc_fraction(transformed, directions)
            channel_score = dc_fraction(transformed_channel, channel)[0]
            for index, score in enumerate(scores):
                alignment_rows.append({
                    "model": model_key, "method": method, "layer": layer,
                    "direction": f"v{index + 1}", "s_i": float(score),
                    "top_magnitude_channel": top_channel,
                })
            alignment_rows.append({
                "model": model_key, "method": method, "layer": layer,
                "direction": "top_magnitude_channel", "s_i": float(channel_score),
                "top_magnitude_channel": top_channel,
            })
        del factor, directions, channel

    tokens = base.prepare_token_chunks(model_id, "train", 0, args.calibration_sequences, args.seq_len, workdir)
    collector = OfflineCollector(model, had, smooth, layers)
    collector.install()
    try:
        act._model_pass(model, tokens, args.batch_size, "E16 offline range/NMSE")
    finally:
        collector.close()
    offline_rows = collector.rows(model_key)
    base.write_csv(result_dir / "e16_offline_per_layer.csv", offline_rows)
    base.write_csv(result_dir / "e16_dc_alignment_per_layer.csv", alignment_rows)
    summary_rows = []
    for method in METHODS:
        for direction in [f"v{i}" for i in range(1, 9)] + ["top_magnitude_channel"]:
            subset = [row["s_i"] for row in alignment_rows if row["method"] == method and row["direction"] == direction]
            summary_rows.append({
                "model": model_key, "method": method, "direction": direction,
                "mean_s_i_across_layers": sum(subset) / len(subset),
            })
    base.write_csv(result_dir / "e16_dc_alignment_summary.csv", summary_rows)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "seed": args.seed,
        "calibration_sequences": args.calibration_sequences, "sequence_length": args.seq_len,
        "offline_methods": ["Hadamard", "SmoothQuant(alpha=.5)+Hadamard"],
        "alignment_methods": list(METHODS), "directions": "top-8 frozen second-moment directions plus top-magnitude channel",
        "s_i": "squared energy in per-group-128 uniform/DC subspace divided by original direction energy",
        "official_duquant": "excluded by later citation-only/no-local-run protocol amendment; E11 DuQuant-style diagnostic retained",
        "paired": "same frozen E11 calibration chunks, signs, statistics, and model",
        "no_tuning": True, "hardware": base.hardware_info(),
    })
    del model, tokens, collector, had, smooth, duquant
    gc.collect()
    torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--model", choices=e11.MODELS, required=True)
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--calibration-sequences", type=int, default=128)
    result.add_argument("--seq-len", type=int, default=2048)
    result.add_argument("--batch-size", type=int, default=1)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
