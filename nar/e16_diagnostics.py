#!/usr/bin/env python3
"""E16 offline SmoothQuant, DC-alignment, and whole-activation diagnostics.

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


class NullSpaceEnergyCollector:
    """Accumulate whole-activation null-space and total energy on fixed rows."""

    def __init__(
        self,
        model: torch.nn.Module,
        signs: dict[tuple[str, int], torch.Tensor],
        duquant: e11.Transform,
        nar_factors: dict[tuple[str, int], act.RotationFactor],
        layers: int,
        dimensions: dict[str, int],
        sample_stride: int,
    ):
        self.model = model
        self.signs = signs
        self.duquant = duquant
        self.nar_factors = nar_factors
        self.layers = layers
        self.dimensions = dimensions
        self.sample_stride = sample_stride
        self.handles: list[Any] = []
        self.values: dict[tuple[str, str, int, str], torch.Tensor] = {}

    def consume(self, site: str, layer: int, value: torch.Tensor) -> None:
        sampled = value.detach()[:, :: self.sample_stride, :].float()
        signs = self.signs[(site, layer)]
        transformed_rows = (
            ("hadamard", act.full_hadamard_rows(sampled, signs)),
            ("duquant_style", self.duquant.activation(site, layer, sampled)),
            ("nar", self.nar_factors[(site, layer)].apply(sampled, signs)),
        )
        for method, transformed in transformed_rows:
            rows = transformed.float().reshape(-1, transformed.shape[-1])
            blocks = rows.reshape(rows.shape[0], -1, 128)
            entries = {
                "null_energy": (blocks.sum(-1).square() / 128).double().sum(),
                "energy": rows.square().double().sum(),
                "rows": torch.tensor(rows.shape[0], device=rows.device, dtype=torch.float64),
            }
            if method in ("hadamard", "nar"):
                dequant, _, _, _ = base.dynamic_asym_int4(rows, 128)
                entries.update({
                    "range_sum": (blocks.amax(-1) - blocks.amin(-1)).double().sum(),
                    "groups": torch.tensor(blocks.numel() // 128, device=rows.device, dtype=torch.float64),
                    "quant_error": (dequant.float() - rows).square().double().sum(),
                })
            for name, number in entries.items():
                key = (method, site, layer, name)
                if key not in self.values:
                    self.values[key] = torch.zeros((), device=rows.device, dtype=torch.float64)
                self.values[key] += number
            del rows, blocks, transformed
        del sampled

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
        for layer, block in enumerate(self.model.model.layers[: self.layers]):
            self.handles.append(block.input_layernorm.register_forward_hook(self.q_hook(layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.down_hook(layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def rows(self, model_key: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for method in ("hadamard", "duquant_style", "nar"):
            for site in ("qkv", "down"):
                for layer in range(self.layers):
                    get = lambda name: float(self.values[(method, site, layer, name)].cpu())
                    energy = get("energy")
                    quantitative = method in ("hadamard", "nar")
                    output.append({
                        "model": model_key,
                        "method": method,
                        "site": site,
                        "layer": layer,
                        "f": get("null_energy") / energy,
                        "mean_group_range": get("range_sum") / get("groups") if quantitative else float("nan"),
                        "nmse": get("quant_error") / energy if quantitative else float("nan"),
                        "rows_used": int(get("rows")),
                        "groups_observed": int(get("groups")) if quantitative else 0,
                        "slots": self.dimensions[site] // 128,
                        "d": self.dimensions[site],
                        "group_size": 128,
                        "sample_stride": self.sample_stride,
                    })
        return output


def run_null_space_energy(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E16 null-space energy requires CUDA")
    if args.seq_len % args.sample_stride:
        raise ValueError("sequence length must be divisible by sample stride")
    workdir = Path(args.workdir).resolve()
    model_id, model_key = act.model_id_and_key(args.model)
    base.setup_logging(workdir, f"e16-null-space-energy-{model_key}")
    result_dir = (
        Path(args.output_dir).resolve() / model_key
        if args.output_dir
        else workdir / "results" / model_key
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    output = result_dir / "e16_null_space_energy_per_layer.csv"
    done = result_dir / "E16_NULL_SPACE_ENERGY_DONE.json"
    if done.exists() and output.exists():
        return

    calibration = e11.calibration_dir(workdir, model_key)
    stats = torch.load(calibration / "channel_stats.pt", map_location="cpu", weights_only=True)
    model = base.load_model(model_id, workdir)
    layers = int(model.config.num_hidden_layers)
    dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
    device = torch.device("cuda")
    signs = {}
    for site_index, (site, n) in enumerate(dimensions.items()):
        for layer in range(layers):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 1000 * layer + 10 * site_index + 128
            )
            signs[(site, layer)] = torch.randint(
                0, 2, (n,), generator=generator, dtype=torch.int64
            ).float().mul_(2).sub_(1).to(device)
    duquant = e11.Transform(
        "duquant_style_g128_asym", model_key, workdir, 0, args.seed,
        layers, dimensions, device, stats
    )
    nar_factors = {
        (site, layer): act.RotationFactor.load(
            _factor_path(workdir, model_key, site, layer), device
        )
        for site in ("qkv", "down")
        for layer in range(layers)
    }
    tokens = base.prepare_token_chunks(
        model_id, "train", 0, args.calibration_sequences, args.seq_len, workdir
    )
    collector = NullSpaceEnergyCollector(
        model, signs, duquant, nar_factors, layers, dimensions, args.sample_stride
    )
    collector.install()
    try:
        act._model_pass(model, tokens, args.batch_size, "E16 whole-activation null-space energy")
    finally:
        collector.close()
    rows = collector.rows(model_key)
    expected_rows = args.calibration_sequences * (args.seq_len // args.sample_stride)
    if any(row["rows_used"] != expected_rows for row in rows):
        raise AssertionError("E16 null-space row-count mismatch")
    base.write_csv(output, rows)
    base.atomic_json(done, {
        "model": model_key,
        "model_id": model_id,
        "seed": args.seed,
        "calibration_split": "train",
        "calibration_offset": 0,
        "calibration_sequences": args.calibration_sequences,
        "sequence_length": args.seq_len,
        "sample_stride": args.sample_stride,
        "rows_used_per_layer_site": expected_rows,
        "group_size": 128,
        "methods": ["Hadamard", "DuQuant-style", "PrismQuant k=max"],
        "metrics": {
            "f": "squared energy in the per-group quantizer null space divided by total transformed activation energy",
            "mean_group_range": "mean per-token per-group max-minus-min",
            "nmse": "global dynamic asymmetric INT4 squared error divided by transformed activation energy",
        },
        "paired": "same calibration chunks, sampled token positions, model, statistics, and fixed transform seed",
        "no_tuning": True,
        "hardware": base.hardware_info(),
    })
    del model, tokens, collector, signs, duquant, nar_factors
    gc.collect()
    torch.cuda.empty_cache()


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
    result.add_argument("--sample-stride", type=int, default=32)
    result.add_argument("--null-space-only", action="store_true")
    result.add_argument("--output-dir", default=None)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.null_space_only:
        run_null_space_energy(arguments)
    else:
        run(arguments)
