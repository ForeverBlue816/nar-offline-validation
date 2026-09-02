#!/usr/bin/env python3
"""E11 fair activation-only baselines, reusing frozen E5 evaluation rows.

Calibration is fixed to the E5 128 WikiText-2 train chunks.  Evaluation adds
only new both-site methods; bf16 and E5 Hadamard/NAR rows are copied verbatim.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from . import extended_experiment as ext
except ImportError:
    import activation_experiments as act
    import experiment as base
    import extended_experiment as ext


LOG = logging.getLogger("nar")
ALPHA = 0.5
BLOCK = 128
TCRIT_DF2_90 = act.TCRIT_DF2_90
MODELS = ("llama32_3b", "llama31_8b")
NAR_SPECS = {
    "nar_b64_kmax": (64, None),
    "nar_b256_kmax": (256, None),
    "nar_b128_k8": (128, 8),
    "nar_b128_k16": (128, 16),
    "nar_b128_k32": (128, 32),
}
NEW_METHODS = (
    "smoothquant_hadamard_g128_asym",
    "duquant_style_g128_asym",
    "hadamard_token_symmetric",
    "hadamard_token_asymmetric",
    *NAR_SPECS,
)
ALL_METHODS = (
    "hadamard_g128_asym",
    "nar_b128_kmax",
    *NEW_METHODS,
)
DUQUANT_UPSTREAM_COMMIT = "d56cfc6fe97c34c0eb100fec82fe439865905679"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def calibration_dir(workdir: Path, model_key: str) -> Path:
    return workdir / "activations" / model_key / "e11_calibration"


def factor_dir(workdir: Path, model_key: str, method: str) -> Path:
    return calibration_dir(workdir, model_key) / "factors" / method


def _key(site: str, layer: int) -> str:
    return f"{site}:{layer}"


class MaxCollector:
    def __init__(self, model: torch.nn.Module, layers: int, dimensions: dict[str, int]):
        self.model = model
        self.layers = layers
        self.maxima = {
            (site, layer): torch.zeros(n, dtype=torch.float32)
            for site, n in dimensions.items() for layer in range(layers)
        }
        self.handles: list[Any] = []

    def consume(self, site: str, layer: int, value: torch.Tensor) -> None:
        flat = value.detach().float().reshape(-1, value.shape[-1])
        maximum = flat.abs().amax(0).cpu()
        self.maxima[(site, layer)] = torch.maximum(self.maxima[(site, layer)], maximum)

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


class VariantEnergyCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        vectors: dict[tuple[str, int], torch.Tensor],
        stride: int,
        layers: int,
    ):
        self.model = model
        self.stride = stride
        self.layers = layers
        self.reflectors: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor, float]] = {}
        self.sums: dict[tuple[str, str, int], torch.Tensor] = {}
        self.counts: dict[tuple[str, str, int], int] = {}
        self.handles: list[Any] = []
        for (site, layer), value in vectors.items():
            n = value.shape[0]
            for method, (b, requested) in NAR_SPECS.items():
                slots = n // b
                rank = slots if requested is None else min(requested, slots)
                refs, active, error = act.reflectors_from_vectors(value[:, :rank].to("cuda"), b)
                key = (method, site, layer)
                self.reflectors[key] = (refs, active, error)
                self.sums[key] = torch.zeros(n, dtype=torch.float64)
                self.counts[key] = 0

    def consume(self, site: str, layer: int, value: torch.Tensor) -> None:
        sampled = value.detach()[:, :: self.stride, :].float().reshape(-1, value.shape[-1])
        for method in NAR_SPECS:
            key = (method, site, layer)
            refs, active, _ = self.reflectors[key]
            transformed = act.apply_reflectors(sampled, refs, active)
            self.sums[key] += transformed.square().sum(0).double().cpu()
            self.counts[key] += transformed.shape[0]
            del transformed

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


def _model_pass(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    label: str,
    capture: MaxCollector | VariantEnergyCollector | None = None,
) -> None:
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], batch_size):
            stop = min(start + batch_size, tokens.shape[0])
            model.model(input_ids=tokens[start:stop].cuda(non_blocking=True), use_cache=False)
            if stop % max(batch_size, 8) == 0 or stop == tokens.shape[0]:
                LOG.info("%s %d/%d sequences", label, stop, tokens.shape[0])


def _weight_maxima(model: torch.nn.Module, layers: int) -> dict[tuple[str, int], torch.Tensor]:
    result: dict[tuple[str, int], torch.Tensor] = {}
    for layer, block in enumerate(model.model.layers[:layers]):
        qkv = (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj)
        result[("qkv", layer)] = torch.stack(
            [module.weight.detach().float().abs().amax(0).cpu() for module in qkv]
        ).amax(0)
        result[("down", layer)] = block.mlp.down_proj.weight.detach().float().abs().amax(0).cpu()
    return result


def _save_factor_variants(
    root: Path,
    dimensions: dict[str, int],
    layers: int,
    energy: VariantEnergyCollector,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    max_error = 0.0
    device = torch.device("cuda")
    for site, n in dimensions.items():
        for layer in range(layers):
            for method, (b, requested) in NAR_SPECS.items():
                slots = n // b
                rank = slots if requested is None else min(requested, slots)
                key = (method, site, layer)
                refs, active, error = energy.reflectors[key]
                means = energy.sums[key] / energy.counts[key]
                source, target = act.balanced_orders(
                    means.clamp_min(0).sqrt().to(device).unsqueeze(0), rank, b
                )
                factor = act.RotationFactor(
                    n, b, refs, active, source.to(device), target.to(device), error
                )
                output = root / "factors" / method / f"{site}_layer_{layer:02d}.pt"
                output.parent.mkdir(parents=True, exist_ok=True)
                factor.save(output, {
                    "site": site,
                    "layer": layer,
                    "method": method,
                    "requested_rank": requested,
                    "realized_rank": rank,
                    "dc_slots": slots,
                    "source": "E11 fixed randomized eigenspace over all calibration tokens",
                    "permutation_energy_rows": energy.counts[key],
                })
                max_error = max(max_error, error)
                rows.append({
                    "model": root.parent.name,
                    "site": site,
                    "layer": layer,
                    "method": method,
                    "b": b,
                    "requested_k": "kmax" if requested is None else requested,
                    "realized_k": rank,
                    "dc_slots": slots,
                    "anchor_error": error,
                    "permutation_energy_rows": energy.counts[key],
                })
                del factor
            LOG.info("E11 factors %s layer=%d/%d", site, layer + 1, layers)
    return rows, max_error


def calibrate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E11 calibration requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    asset_workdir = Path(args.asset_workdir).resolve() if args.asset_workdir else workdir
    model_id, model_key = act.model_id_and_key(args.model)
    if model_key not in MODELS:
        raise ValueError(model_key)
    base.setup_logging(workdir, f"e11-calibrate-{model_key}")
    root = calibration_dir(workdir, model_key)
    done = root / "DONE.json"
    if done.exists():
        LOG.info("E11 calibration exists: %s", done)
        return
    base.seed_everything(args.seed)
    tokens = base.prepare_token_chunks(
        model_id, "train", 0, args.calibration_sequences, args.seq_len, asset_workdir
    )
    model = base.load_model(model_id, asset_workdir)
    total_layers = int(model.config.num_hidden_layers)
    layers = total_layers if args.max_layers is None else min(args.max_layers, total_layers)
    if layers != total_layers:
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[:layers]))
        model.config.num_hidden_layers = layers
    dimensions = {
        "qkv": int(model.config.hidden_size),
        "down": int(model.config.intermediate_size),
    }
    bases: dict[tuple[str, int], torch.Tensor] = {}
    for site, n in dimensions.items():
        width = n // 64 + args.oversample
        for layer in range(layers):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 100_000 * (site == "down") + layer
            )
            omega = torch.randn((n, width), generator=generator, dtype=torch.float64)
            bases[(site, layer)] = torch.linalg.qr(omega, mode="reduced").Q.float()
    final_cq: dict[tuple[str, int], torch.Tensor] = {}
    traces: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    maxima: dict[tuple[str, int], torch.Tensor] = {}
    for pass_index in range(3):
        sketch = act.SketchCollector(model, bases, trace=pass_index == 2)
        sketch.install()
        capture = None
        if pass_index == 0:
            capture = MaxCollector(model, layers, dimensions)
            capture.install()
        try:
            _model_pass(
                model, tokens, args.batch_size,
                f"E11 covariance pass {pass_index + 1}/3",
                capture,
            )
        finally:
            sketch.close()
            if capture is not None:
                capture.close()
                maxima = capture.maxima
        for key, result in sketch.results.items():
            normalized = result / sketch.counts[key]
            if pass_index < 2:
                bases[key] = torch.linalg.qr(normalized, mode="reduced").Q.float()
            else:
                final_cq[key] = normalized
                traces[key] = sketch.traces[key] / sketch.counts[key]
                counts[key] = sketch.counts[key]
        del sketch, capture
        gc.collect()
        torch.cuda.empty_cache()
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    eig_rows: list[dict[str, Any]] = []
    for key, q in bases.items():
        cq = final_cq[key]
        small = q.double().T @ cq.double()
        small = (small + small.T) / 2
        values, u = torch.linalg.eigh(small)
        rank = q.shape[0] // 64
        order = torch.argsort(values, descending=True)[:rank]
        values = values[order].clamp_min(0)
        u = u[:, order]
        v = q.double() @ u
        residual = cq.double() @ u - v * values.unsqueeze(0)
        relative = residual.norm(dim=0) / values.clamp_min(torch.finfo(torch.float64).tiny)
        vectors[key] = v.float().cpu()
        site, layer = key
        for index in range(rank):
            eig_rows.append({
                "model": model_key,
                "site": site,
                "layer": layer,
                "rank": index + 1,
                "eigenvalue": float(values[index]),
                "fraction_total_energy": float(values[index]) / traces[key],
                "cumulative_fraction_total_energy": float(values[: index + 1].sum()) / traces[key],
                "relative_ritz_residual": float(relative[index]),
            })
    weight_max = _weight_maxima(model, layers)
    scales: dict[str, torch.Tensor] = {}
    stats: dict[str, torch.Tensor] = {}
    stat_rows: list[dict[str, Any]] = []
    for site, n in dimensions.items():
        for layer in range(layers):
            key = (site, layer)
            act_max = maxima[key].float()
            w_max = weight_max[key].float()
            scale = (
                act_max.clamp_min(1e-8).pow(ALPHA)
                / w_max.clamp_min(1e-8).pow(1.0 - ALPHA)
            )
            scales[_key(site, layer)] = scale
            stats[_key(site, layer)] = act_max
            stat_rows.append({
                "model": model_key,
                "site": site,
                "layer": layer,
                "channels": n,
                "activation_absmax_mean": float(act_max.mean()),
                "activation_absmax_max": float(act_max.max()),
                "weight_absmax_mean": float(w_max.mean()),
                "weight_absmax_max": float(w_max.max()),
                "smooth_scale_min": float(scale.min()),
                "smooth_scale_max": float(scale.max()),
                "smooth_scale_mean": float(scale.mean()),
            })
    base.atomic_torch_save(root / "channel_stats.pt", {
        "activation_absmax": stats,
        "smoothquant_scale": scales,
        "alpha": ALPHA,
    })
    del weight_max, maxima, bases, final_cq
    gc.collect()
    torch.cuda.empty_cache()
    energy = VariantEnergyCollector(model, vectors, args.sample_stride, layers)
    energy.install()
    try:
        _model_pass(model, tokens, args.batch_size, "E11 permutation-energy pass 1/1")
    finally:
        energy.close()
    factor_rows, max_anchor_error = _save_factor_variants(root, dimensions, layers, energy)
    del energy, model, tokens, vectors
    gc.collect()
    torch.cuda.empty_cache()
    result_dir = workdir / "results" / model_key
    base.write_csv(result_dir / "e11_calibration_stats.csv", stat_rows)
    base.write_csv(result_dir / "e11_calibration_eigenspace.csv", eig_rows)
    base.write_csv(result_dir / "e11_factor_audit.csv", factor_rows)
    base.atomic_json(done, {
        "model": model_key,
        "model_id": model_id,
        "layers": layers,
        "dimensions": dimensions,
        "calibration_split": "train",
        "calibration_offset": 0,
        "calibration_sequences": args.calibration_sequences,
        "sequence_length": args.seq_len,
        "permutation_energy_stride": args.sample_stride,
        "eigenspace_rank": {site: n // 64 for site, n in dimensions.items()},
        "eigenspace_retention": "vectors are transient; eigenvalues, energy fractions, and Ritz residuals are retained in CSV",
        "oversample": args.oversample,
        "power_iterations": 1,
        "covariance_passes": 3,
        "smoothquant_alpha": ALPHA,
        "max_anchor_error": max_anchor_error,
        "duquant_reference_commit": DUQUANT_UPSTREAM_COMMIT,
        "hardware": base.hardware_info(),
    })


def _random_signs(n: int, seed_index: int, base_seed: int, layer: int, site: str, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        act._seed(base_seed, seed_index, layer, site)
    )
    return torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1).to(device)


def _zigzag_permutation(scores: torch.Tensor, b: int) -> torch.Tensor:
    order = sorted(range(scores.numel()), key=lambda index: (-float(scores[index]), index))
    groups = [[] for _ in range(scores.numel() // b)]
    current = 0
    upward = True
    for index in order:
        groups[current].append(index)
        if upward:
            current += 1
            if current == len(groups):
                current -= 1
                upward = False
        else:
            current -= 1
            if current == -1:
                current += 1
                upward = True
    for group in groups:
        group.sort(key=lambda index: (-float(scores[index]), index))
    return torch.tensor([index for group in groups for index in group], dtype=torch.long)


def _duquant_blocks(
    scores: torch.Tensor,
    b: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = _zigzag_permutation(scores, b)
    permuted_scores = scores[permutation].reshape(-1, b)
    groups = permuted_scores.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn((groups, b - 1, b - 1), generator=generator, dtype=torch.float32).to(device)
    random_q = torch.linalg.qr(random, mode="reduced").Q
    hadamard = base.hadamard(b, dtype=torch.float32).to(device)
    uniform = hadamard[0]
    complement = random_q @ hadamard[1:]
    rotations = torch.cat((uniform.expand(groups, 1, b), complement), dim=1)
    peaks = permuted_scores.argmax(1).to(device)
    group_index = torch.arange(groups, device=device)
    first = rotations[:, 0].clone()
    peak_rows = rotations[group_index, peaks].clone()
    rotations[:, 0] = peak_rows
    rotations[group_index, peaks] = first
    error = (rotations @ rotations.transpose(1, 2) - torch.eye(b, device=device)).abs().max()
    if float(error) > 2e-5:
        raise AssertionError(f"DuQuant-style block orthogonality error {float(error)}")
    return permutation.to(device), rotations



@dataclass
class Transform:
    method: str
    model_key: str
    workdir: Path
    seed_index: int
    base_seed: int
    layers: int
    dimensions: dict[str, int]
    device: torch.device
    stats: dict[str, Any]

    def __post_init__(self) -> None:
        self.signs: dict[tuple[str, int], torch.Tensor] = {}
        self.factors: dict[tuple[str, int], act.RotationFactor] = {}
        self.permutations: dict[tuple[str, int], torch.Tensor] = {}
        self.blocks: dict[tuple[str, int], torch.Tensor] = {}
        for site, n in self.dimensions.items():
            for layer in range(self.layers):
                if self.method != "duquant_style_g128_asym":
                    self.signs[(site, layer)] = _random_signs(
                        n, self.seed_index, self.base_seed, layer, site, self.device
                    )
                if self.method in NAR_SPECS:
                    self.factors[(site, layer)] = act.RotationFactor.load(
                        factor_dir(self.workdir, self.model_key, self.method)
                        / f"{site}_layer_{layer:02d}.pt",
                        self.device,
                    )
                if self.method == "duquant_style_g128_asym":
                    scores = self.stats["activation_absmax"][_key(site, layer)]
                    permutation, blocks = _duquant_blocks(
                        scores, BLOCK,
                        self.base_seed + self.seed_index + 100_000 * (site == "down") + 1000 * layer,
                        self.device,
                    )
                    self.permutations[(site, layer)] = permutation
                    self.blocks[(site, layer)] = blocks

    def _orthogonal(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        if self.method in NAR_SPECS:
            return self.factors[(site, layer)].apply(value, self.signs[(site, layer)])
        if self.method == "duquant_style_g128_asym":
            original_shape = value.shape
            rows = value.float().reshape(-1, value.shape[-1])
            rows = rows[:, self.permutations[(site, layer)]]
            blocks = self.blocks[(site, layer)]
            rows = torch.einsum("ngi,gij->ngj", rows.reshape(-1, blocks.shape[0], BLOCK), blocks)
            return rows.reshape(original_shape)
        return act.full_hadamard_rows(value.float(), self.signs[(site, layer)])

    def activation(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        x = value.float()
        if self.method == "smoothquant_hadamard_g128_asym":
            scale = self.stats["smoothquant_scale"][_key(site, layer)].to(self.device)
            x = x / scale
        return self._orthogonal(site, layer, x)

    def weight(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        w = value.float()
        if self.method == "smoothquant_hadamard_g128_asym":
            scale = self.stats["smoothquant_scale"][_key(site, layer)].to(self.device)
            w = w * scale
        return self._orthogonal(site, layer, w)


def _symmetric_int4(value: torch.Tensor) -> torch.Tensor:
    x = value.float()
    maximum = x.abs().amax(dim=-1, keepdim=True)
    raw_scale = maximum / 7.0
    scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale)).to(torch.float16).float()
    q = torch.round(x / scale).clamp_(-8, 7)
    return (q * scale).to(value.dtype)


class Hooks:
    def __init__(self, model: torch.nn.Module, transform: Transform):
        self.model = model
        self.transform = transform
        self.handles: list[Any] = []

    def quantize(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        rotated = self.transform.activation(site, layer, value)
        method = self.transform.method
        if method == "hadamard_token_symmetric":
            dequant = _symmetric_int4(rotated)
        elif method == "hadamard_token_asymmetric":
            dequant, _, _, _ = base.dynamic_asym_int4(rotated, rotated.shape[-1])
        else:
            b = NAR_SPECS[method][0] if method in NAR_SPECS else BLOCK
            dequant, _, _, _ = base.dynamic_asym_int4(rotated, b)
        return dequant.to(value.dtype)

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers[: self.transform.layers]):
            self.handles.append(
                block.input_layernorm.register_forward_hook(
                    lambda _m, _i, output, layer=layer: self.quantize("qkv", layer, output)
                )
            )
            self.handles.append(
                block.mlp.down_proj.register_forward_pre_hook(
                    lambda _m, inputs, layer=layer: (self.quantize("down", layer, inputs[0]),)
                )
            )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class WeightManager:
    def __init__(self, model: torch.nn.Module, layers: int):
        self.modules: dict[tuple[str, int], list[torch.nn.Linear]] = {}
        self.originals: dict[int, torch.Tensor] = {}
        for layer, block in enumerate(model.model.layers[:layers]):
            qkv = [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]
            down = [block.mlp.down_proj]
            self.modules[("qkv", layer)] = qkv
            self.modules[("down", layer)] = down
            for module in qkv + down:
                self.originals[id(module)] = module.weight.detach().to("cpu", dtype=torch.bfloat16).clone()

    def restore(self) -> None:
        with torch.no_grad():
            for modules in self.modules.values():
                for module in modules:
                    module.weight.copy_(self.originals[id(module)].to(module.weight.device))

    def apply(self, transform: Transform, row_batch: int) -> float:
        maximum = 0.0
        with torch.no_grad():
            for (site, layer), modules in self.modules.items():
                for module in modules:
                    original = self.originals[id(module)]
                    chunks = []
                    for start in range(0, original.shape[0], row_batch):
                        chunk = original[start:start + row_batch].to(module.weight.device)
                        chunks.append(transform.weight(site, layer, chunk).to(torch.bfloat16))
                    folded = torch.cat(chunks, 0)
                    module.weight.copy_(folded)
                    generator = torch.Generator(device="cpu").manual_seed(
                        911 + layer + 10_000 * (site == "down")
                    )
                    probe = torch.randn((2, original.shape[1]), generator=generator).to(module.weight.device)
                    reference = probe @ original[: min(16, original.shape[0])].to(probe.device).float().T
                    transformed_probe = transform.activation(site, layer, probe)
                    observed = transformed_probe @ folded[: min(16, folded.shape[0])].float().T
                    relative = float((observed - reference).norm() / reference.norm().clamp_min(1e-30))
                    maximum = max(maximum, relative)
                    del chunks, folded, probe, reference, transformed_probe, observed
        return maximum


def _metadata_bits(method: str, n: int) -> float:
    if method == "bf16":
        return 16.0
    if method == "hadamard_token_symmetric":
        return 4.0 + 16.0 / n
    if method == "hadamard_token_asymmetric":
        return 4.0 + 32.0 / n
    b = NAR_SPECS[method][0] if method in NAR_SPECS else BLOCK
    return 4.0 + 32.0 / b


def _reuse_e5(workdir: Path, model_key: str) -> list[dict[str, Any]]:
    source = base.read_csv(workdir / "results" / model_key / "e5_per_sequence.csv")
    rows: list[dict[str, Any]] = []
    for row in source:
        if row["site"] == "none" and row["method"] == "bf16":
            method = "bf16"
        elif row["site"] == "both" and row["method"] == "hadamard":
            method = "hadamard_g128_asym"
        elif row["site"] == "both" and row["method"] == "nar":
            method = "nar_b128_kmax"
        else:
            continue
        rows.append({
            "model": model_key,
            "method": method,
            "seed": int(row["seed"]),
            "sequence": int(row["sequence"]),
            "nll": float(row["nll"]),
            "tokens_scored": int(row["tokens_scored"]),
            "source": "reused_verbatim_from_e5",
            "weight_fold_max_relative_error": float(row["weight_fold_max_relative_error"]),
        })
    return rows


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    half = TCRIT_DF2_90 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def summarize(
    rows: list[dict[str, Any]],
    dimensions: dict[str, int],
) -> list[dict[str, Any]]:
    bf = [float(row["nll"]) for row in rows if row["method"] == "bf16"]
    bf_ppl = math.exp(float(np.mean(bf)))
    seeds = sorted({int(row["seed"]) for row in rows if int(row["seed"]) >= 0})
    ppls: dict[tuple[str, int], float] = {}
    for method in ALL_METHODS:
        for seed in seeds:
            values = [
                float(row["nll"]) for row in rows
                if row["method"] == method and int(row["seed"]) == seed
            ]
            if len(values) != len(bf):
                raise RuntimeError(f"incomplete E11 method={method} seed={seed}: {len(values)}")
            ppls[(method, seed)] = math.exp(float(np.mean(values)))
    output = [{
        "model": rows[0]["model"],
        "method": "bf16",
        "seeds": 1,
        "mean_ppl": bf_ppl,
        "ppl_delta_vs_bf16": 0.0,
        "paired_ppl_delta_vs_hadamard": math.nan,
        "paired_90ci_low_vs_hadamard": math.nan,
        "paired_90ci_high_vs_hadamard": math.nan,
        "paired_ppl_delta_vs_nar": math.nan,
        "paired_90ci_low_vs_nar": math.nan,
        "paired_90ci_high_vs_nar": math.nan,
        "effective_bits_qkv": 16.0,
        "effective_bits_down": 16.0,
        "metadata_formula": "bf16 value; no quantization metadata",
        "hadamard_to_bf16_gap_recovered": math.nan,
    }]
    for method in ALL_METHODS:
        values = np.asarray([ppls[(method, seed)] for seed in seeds])
        versus_had = np.asarray([
            ppls[(method, seed)] - ppls[("hadamard_g128_asym", seed)] for seed in seeds
        ])
        versus_nar = np.asarray([
            ppls[(method, seed)] - ppls[("nar_b128_kmax", seed)] for seed in seeds
        ])
        had_mean, had_low, had_high = _mean_ci(versus_had)
        nar_mean, nar_low, nar_high = _mean_ci(versus_nar)
        gaps = np.asarray([
            (
                ppls[("hadamard_g128_asym", seed)] - ppls[(method, seed)]
            ) / (
                ppls[("hadamard_g128_asym", seed)] - bf_ppl
            ) for seed in seeds
        ])
        if method == "hadamard_token_symmetric":
            formula = "4 + 16/n (one fp16 scale per token)"
        elif method == "hadamard_token_asymmetric":
            formula = "4 + 32/n (one fp16 scale + fp16 zero-point per token)"
        else:
            b = NAR_SPECS[method][0] if method in NAR_SPECS else BLOCK
            formula = f"4 + 32/{b} (one fp16 scale + fp16 zero-point per group)"
        output.append({
            "model": rows[0]["model"],
            "method": method,
            "seeds": len(seeds),
            "mean_ppl": float(values.mean()),
            "ppl_delta_vs_bf16": float(values.mean() - bf_ppl),
            "paired_ppl_delta_vs_hadamard": had_mean,
            "paired_90ci_low_vs_hadamard": had_low,
            "paired_90ci_high_vs_hadamard": had_high,
            "paired_ppl_delta_vs_nar": nar_mean,
            "paired_90ci_low_vs_nar": nar_low,
            "paired_90ci_high_vs_nar": nar_high,
            "effective_bits_qkv": _metadata_bits(method, dimensions["qkv"]),
            "effective_bits_down": _metadata_bits(method, dimensions["down"]),
            "metadata_formula": formula,
            "hadamard_to_bf16_gap_recovered": float(gaps.mean()),
        })
    return output


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E11 evaluation requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    asset_workdir = Path(args.asset_workdir).resolve() if args.asset_workdir else workdir
    model_id, model_key = act.model_id_and_key(args.model)
    if model_key not in MODELS:
        raise ValueError(model_key)
    base.setup_logging(workdir, f"e11-evaluate-{model_key}")
    result_dir = workdir / "results" / model_key
    done = result_dir / "E11_DONE.json"
    if done.exists():
        LOG.info("E11 result exists: %s", done)
        return
    cal_meta = _json(calibration_dir(workdir, model_key) / "DONE.json")
    stats = torch.load(
        calibration_dir(workdir, model_key) / "channel_stats.pt",
        map_location="cpu",
        weights_only=True,
    )
    tokens = base.prepare_token_chunks(model_id, "test", 0, args.eval_sequences, args.seq_len, asset_workdir)
    model = base.load_model(model_id, asset_workdir)
    total_layers = int(model.config.num_hidden_layers)
    layers = total_layers if args.max_layers is None else min(args.max_layers, total_layers)
    if layers != int(cal_meta["layers"]):
        raise RuntimeError("E11 calibration/evaluation layer mismatch")
    if layers != total_layers:
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[:layers]))
        model.config.num_hidden_layers = layers
    dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
    formal_reuse = layers == total_layers and args.eval_sequences == 64
    reused = _reuse_e5(workdir, model_key) if formal_reuse else []
    if not formal_reuse:
        with torch.inference_mode():
            losses = act.evaluate_nlls(model, tokens, f"{model_key} E11 smoke bf16")
        reused.extend({
            "model": model_key, "method": "bf16", "seed": -1, "sequence": index,
            "nll": loss, "tokens_scored": args.seq_len - 1, "source": "smoke",
            "weight_fold_max_relative_error": 0.0,
        } for index, loss in enumerate(losses))
        # Smoke substitutes will be populated by explicit methods below.
    partial = result_dir / "e11_per_sequence.partial.csv"
    rows = list(base.read_csv(partial)) if partial.exists() else reused
    completed = {(str(row["method"]), int(row["seed"])) for row in rows}
    manager = WeightManager(model, layers)
    audit_rows: list[dict[str, Any]] = []
    methods = list(NEW_METHODS)
    for method in methods:
        for seed_index in range(args.seeds):
            seed = args.seed + seed_index
            if (method, seed) in completed:
                continue
            transform_method = method
            transform = Transform(
                transform_method, model_key, workdir, seed_index, args.seed,
                layers, dimensions, torch.device("cuda"), stats,
            )
            manager.restore()
            fold_error = manager.apply(transform, args.weight_row_batch)
            hooks = Hooks(model, transform)
            hooks.install()
            try:
                losses = act.evaluate_nlls(model, tokens, f"{model_key} E11 {method} seed={seed}")
            finally:
                hooks.close()
            rows.extend({
                "model": model_key,
                "method": method,
                "seed": seed,
                "sequence": index,
                "nll": loss,
                "tokens_scored": args.seq_len - 1,
                "source": "e11_new",
                "weight_fold_max_relative_error": fold_error,
            } for index, loss in enumerate(losses))
            audit_rows.append({
                "model": model_key,
                "method": method,
                "seed": seed,
                "max_relative_weight_fold_error": fold_error,
            })
            base.write_csv(partial, rows)
            LOG.info("E11 complete %s seed=%d fold_error=%.6g", method, seed, fold_error)
            del transform
            gc.collect()
            torch.cuda.empty_cache()
    manager.restore()
    if not formal_reuse:
        LOG.info("smoke mode: skipping formal paired summary")
        base.write_csv(result_dir / "e11_smoke_per_sequence.csv", rows)
        done_payload = {"model": model_key, "smoke": True, "methods": methods, "hardware": base.hardware_info()}
        base.atomic_json(done, done_payload)
        partial.unlink(missing_ok=True)
        return
    final_rows = [{key: value for key, value in row.items()} for row in rows]
    summary = summarize(final_rows, dimensions)
    base.write_csv(result_dir / "e11_per_sequence.csv", final_rows)
    base.write_csv(result_dir / "e11_summary.csv", summary)
    if audit_rows:
        base.write_csv(result_dir / "e11_weight_fold_audit.csv", audit_rows)
    baseline_rows = [
        row for row in summary
        if row["method"] in ("smoothquant_hadamard_g128_asym", "duquant_style_g128_asym")
    ]
    matched = []
    for row in baseline_rows:
        low = float(row["paired_90ci_low_vs_nar"])
        high = float(row["paired_90ci_high_vs_nar"])
        if low <= 0.0:
            matched.append(row["method"])
    base.atomic_json(done, {
        "model": model_key,
        "model_id": model_id,
        "site": "both",
        "activation_only": True,
        "eval_split": "test",
        "eval_sequences": args.eval_sequences,
        "sequence_length": args.seq_len,
        "seeds": args.seeds,
        "base_seed": args.seed,
        "reused_rows": ["bf16", "E5 both-site Hadamard", "E5 both-site NAR"],
        "new_methods": list(NEW_METHODS),
        "smoothquant": {
            "alpha": ALPHA,
            "formula": "s_c=max|x_c|^0.5/max|w_c|^0.5; x'=x/s, W'=W*s, then random-sign Hadamard",
            "no_sweep": True,
        },
        "duquant_style": {
            "reference_commit": DUQUANT_UPSTREAM_COMMIT,
            "implemented": "official zigzag distribution by calibration channel absmax, then one blockwise rotation whose largest channel row is uniform and whose complement is seeded random orthogonal",
            "difference_from_official": "no global greedy multi-step prefix and no second post-permutation rotation",
            "difference_from_nar": "single greedy channel per block rather than top-k second-moment eigen-directions; no explicit zero-point/DC alignment",
        },
        "metadata": {
            "group_asymmetric": "4 + (16 scale bits + 16 zero-point bits)/group_size",
            "token_symmetric": "4 + 16/n",
            "token_asymmetric": "4 + 32/n",
        },
        "stop_condition_matched_methods": sorted(set(matched)),
        "stop_before_e14": bool(matched),
        "paired_ci": "two-sided 90% Student-t over three seed-level PPL differences, df=2",
        "no_tuning": True,
        "hardware": base.hardware_info(),
    })
    partial.unlink(missing_ok=True)


def decision(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    summaries = []
    completions = []
    for model in MODELS:
        result_dir = workdir / "results" / model
        summaries.extend(base.read_csv(result_dir / "e11_summary.csv"))
        completions.append(_json(result_dir / "E11_DONE.json"))
        # Reconstruct the audit from the durable per-sequence rows.  This keeps
        # the audit complete when evaluation resumed after an interrupted job.
        audit = {}
        for row in base.read_csv(result_dir / "e11_per_sequence.csv"):
            method = str(row["method"])
            if method == "bf16" or row.get("weight_fold_max_relative_error", "") == "":
                continue
            key = (method, int(row["seed"]))
            audit[key] = {
                "model": model,
                "method": method,
                "seed": int(row["seed"]),
                "source": str(row.get("source", "")),
                "max_relative_weight_fold_error": float(row["weight_fold_max_relative_error"]),
            }
        base.write_csv(result_dir / "e11_weight_fold_audit.csv", list(audit.values()))
    base.write_csv(workdir / "results" / "e11_headline.csv", summaries)
    matched = {
        model["model"]: model["stop_condition_matched_methods"]
        for model in completions if model["stop_condition_matched_methods"]
    }
    base.atomic_json(workdir / "results" / "decision_e11.json", {
        "stop_before_e14": bool(matched),
        "matched_or_better_than_nar_within_paired_90ci": matched,
        "models": list(MODELS),
        "rule": "stop before E14 if SmoothQuant+Hadamard or DuQuant-style has baseline-minus-NAR CI with lower bound <= 0",
        "e5_rows_reused_without_rerun": True,
        "no_tuning": True,
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--asset-workdir", help="optional shared model/token cache root")
    sub = result.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--model", choices=MODELS, required=True)
    cal.add_argument("--calibration-sequences", type=int, default=128)
    cal.add_argument("--seq-len", type=int, default=2048)
    cal.add_argument("--batch-size", type=int, default=1)
    cal.add_argument("--oversample", type=int, default=16)
    cal.add_argument("--sample-stride", type=int, default=32)
    cal.add_argument("--max-layers", type=int)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", choices=MODELS, required=True)
    ev.add_argument("--eval-sequences", type=int, default=64)
    ev.add_argument("--seq-len", type=int, default=2048)
    ev.add_argument("--seeds", type=int, default=3)
    ev.add_argument("--weight-row-batch", type=int, default=512)
    ev.add_argument("--max-layers", type=int)
    sub.add_parser("decision")
    return result


def main() -> None:
    args = parser().parse_args()
    {"calibrate": calibrate, "evaluate": evaluate, "decision": decision}[args.command](args)


if __name__ == "__main__":
    main()
