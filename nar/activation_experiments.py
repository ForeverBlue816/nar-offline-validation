#!/usr/bin/env python3
"""Activation-only NAR validation: E5 runtime PPL and E6 online cost.

K is deliberately out of scope here.  Calibration and evaluation choices are
fixed before results are observed; completed E1/E2/E1c artifacts are read-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

try:
    from . import experiment as base
    from . import extended_experiment as ext
except ImportError:
    import experiment as base
    import extended_experiment as ext


LOG = logging.getLogger("nar")
GROUP_SIZE = 128
MODEL_IDS = {
    "llama32_3b": "unsloth/Llama-3.2-3B",
    "llama32_1b": "unsloth/Llama-3.2-1B",
    "llama31_8b": "unsloth/Meta-Llama-3.1-8B",
    # E18 v2 / E19 use the base checkpoint, never the post-trained Qwen/Qwen3-8B.
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
}
SITES = ("qkv", "down")
EVAL_SITES = ("qkv_only", "both", "down_only")
METHODS = ("identity", "hadamard", "nar")
TCRIT_DF2_90 = 2.919985580353725


def model_id_and_key(value: str) -> tuple[str, str]:
    if value in MODEL_IDS:
        return MODEL_IDS[value], value
    for key, model_id in MODEL_IDS.items():
        if value == model_id:
            return model_id, key
    raise ValueError(f"unsupported frozen model: {value}")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict[str, str]]:
    return base.read_csv(path) if path.exists() else []


def _seed(base_seed: int, seed_index: int, layer: int, site: str) -> int:
    return base_seed + seed_index + 1000 * layer + (0 if site == "qkv" else 100_000) + GROUP_SIZE


def _gf27_mul(a: int, b: int) -> int:
    """Multiply in GF(3^3), modulus x^3 + 2x + 1."""
    ac = [(a // (3**i)) % 3 for i in range(3)]
    bc = [(b // (3**i)) % 3 for i in range(3)]
    prod = [0] * 5
    for i in range(3):
        for j in range(3):
            prod[i + j] = (prod[i + j] + ac[i] * bc[j]) % 3
    # x^3 = x + 2 and x^4 = x^2 + 2x over this modulus.
    for degree in (4, 3):
        coefficient = prod[degree] % 3
        if not coefficient:
            continue
        prod[degree] = 0
        prod[degree - 3] = (prod[degree - 3] + 2 * coefficient) % 3
        prod[degree - 2] = (prod[degree - 2] + coefficient) % 3
    return prod[0] + 3 * prod[1] + 9 * prod[2]


def _gf27_sub(a: int, b: int) -> int:
    coefficients = [((a // (3**i)) - (b // (3**i))) % 3 for i in range(3)]
    return coefficients[0] + 3 * coefficients[1] + 9 * coefficients[2]


def paley_hadamard_28(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    residues = {_gf27_mul(value, value) for value in range(1, 27)}
    if len(residues) != 13:
        raise AssertionError("GF(27) quadratic-residue construction failed")
    core = torch.empty((27, 27), device=device, dtype=dtype)
    for row in range(27):
        for column in range(27):
            delta = _gf27_sub(row, column)
            core[row, column] = 0 if delta == 0 else (1 if delta in residues else -1)
    matrix = torch.ones((28, 28), device=device, dtype=dtype)
    matrix[1:, 1:] = core - torch.eye(27, device=device, dtype=dtype)
    error = (matrix @ matrix.T - 28 * torch.eye(28, device=device, dtype=dtype)).abs().max()
    if float(error) > 1e-4:
        raise AssertionError(f"invalid Paley-28 Hadamard: {float(error)}")
    return matrix / math.sqrt(28)


def full_hadamard_rows(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    quotient, remainder = divmod(n, 28)
    if not remainder and quotient >= 1 and not quotient & (quotient - 1):
        signed = x * signs
        factored = ext._fast_walsh_hadamard(signed.reshape(-1, 28, quotient))
        h28 = paley_hadamard_28(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h28.T).transpose(1, 2).reshape_as(x)
    quotient, remainder = divmod(n, 12)
    if not remainder and quotient >= 1 and not quotient & (quotient - 1):
        signed = x * signs
        factored = ext._fast_walsh_hadamard(signed.reshape(-1, 12, quotient))
        h12 = ext._paley_hadamard_12(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h12.T).transpose(1, 2).reshape_as(x)
    return ext._full_hadamard_rows(x, signs)


def balanced_orders(x_after_g: torch.Tensor, rank: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = x_after_g.shape[-1]
    groups = n // b
    energies = x_after_g.square().mean(0).double().cpu()
    absorbed_sources = [index * b for index in range(rank)]
    remaining = [index for index in range(n) if index not in absorbed_sources]
    fillers = sorted(remaining, key=lambda index: (float(energies[index]), index))[: groups - rank]
    anchor_sources = absorbed_sources + fillers
    residual_sources = [index for index in remaining if index not in fillers]
    residual_sources.sort(key=lambda index: (-float(energies[index]), index))
    residual_energies = [max(0.0, float(energies[index])) for index in residual_sources]
    target = [group * b for group in range(groups)] + base._balanced_target_slots(residual_energies, groups, b)
    source = anchor_sources + residual_sources
    if len(source) != n or len(set(source)) != n or len(set(target)) != n:
        raise AssertionError("invalid calibrated permutation")
    return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def apply_reflectors(x: torch.Tensor, reflectors: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    output = x.float()
    for index in range(reflectors.shape[0]):
        if bool(active[index]):
            vector = reflectors[index]
            output = output - 2 * (output @ vector).unsqueeze(-1) * vector
    return output


def reflectors_from_vectors(vectors: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    refs, error = ext._householders_to_anchors(vectors.float(), b)
    active = torch.tensor([item is not None for item in refs], dtype=torch.bool)
    stacked = torch.stack([
        item if item is not None else torch.zeros(vectors.shape[0], dtype=torch.float32, device=vectors.device)
        for item in refs
    ]).float()
    return stacked, active.to(vectors.device), error


@dataclass
class RotationFactor:
    n: int
    b: int
    reflectors: torch.Tensor
    active: torch.Tensor
    source_order: torch.Tensor
    target_order: torch.Tensor
    anchor_error: float

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "RotationFactor":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            n=int(payload["n"]), b=int(payload["b"]),
            reflectors=payload["reflectors"].float().to(device),
            active=payload["active"].bool().to(device),
            source_order=payload["source_order"].long().to(device),
            target_order=payload["target_order"].long().to(device),
            anchor_error=float(payload["anchor_error"]),
        )

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "n": self.n, "b": self.b,
            "reflectors": self.reflectors.float().cpu(),
            "active": self.active.bool().cpu(),
            "source_order": self.source_order.long().cpu(),
            "target_order": self.target_order.long().cpu(),
            "anchor_error": self.anchor_error,
        }
        if extra:
            payload.update(extra)
        base.atomic_torch_save(path, payload)

    def apply(self, x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        rows = x.float().reshape(-1, self.n)
        rows = apply_reflectors(rows, self.reflectors, self.active)
        permuted = torch.empty_like(rows)
        permuted[:, self.target_order] = rows[:, self.source_order]
        signed = (permuted * signs).reshape(-1, self.n // self.b, self.b)
        return ext._fast_walsh_hadamard(signed).reshape(original_shape)


def factor_from_vectors(vectors: torch.Tensor, calibration_rows: torch.Tensor, b: int) -> RotationFactor:
    reflectors, active, error = reflectors_from_vectors(vectors, b)
    after_g = apply_reflectors(calibration_rows.float(), reflectors, active)
    source, target = balanced_orders(after_g, vectors.shape[1], b)
    return RotationFactor(
        n=vectors.shape[0], b=b, reflectors=reflectors, active=active,
        source_order=source.to(vectors.device), target_order=target.to(vectors.device),
        anchor_error=error,
    )


def factor_dir(workdir: Path, model_key: str) -> Path:
    return workdir / "activations" / model_key / "activation_factors"


def build_factors_from_e1c(workdir: Path, model_key: str, stride: int) -> None:
    if model_key != "llama32_3b":
        raise ValueError("only the 3B model has frozen E1c dumps")
    output = factor_dir(workdir, model_key)
    done = output / "DONE.json"
    if done.exists():
        LOG.info("activation factor checkpoint exists: %s", done)
        return
    wide = workdir / "activations" / model_key / "wide_cal_a"
    meta = _json(wide / "DONE.json")
    eig_dir = wide / "analysis" / "eigenspaces"
    device = torch.device("cuda")
    output.mkdir(parents=True, exist_ok=True)
    errors: list[float] = []
    for site in SITES:
        source_site = "q_input" if site == "qkv" else "down_input"
        n = int(meta["hidden_size"] if site == "qkv" else meta["intermediate_size"])
        for layer in range(int(meta["num_layers"])):
            path = output / f"{site}_layer_{layer:02d}.pt"
            if path.exists():
                continue
            eig = torch.load(eig_dir / f"{source_site}_layer_{layer:02d}.pt", map_location="cpu", weights_only=True)
            vectors = eig["vectors"].float().to(device)
            mmap = ext._open_site(wide, meta, source_site, layer)
            calibration = ext._sample_site_tokens(mmap, stride, device)
            factor = factor_from_vectors(vectors, calibration, GROUP_SIZE)
            factor.save(path, {"source": "frozen E1c", "site": site, "layer": layer})
            errors.append(factor.anchor_error)
            LOG.info("factor %s layer=%d/%d", site, layer + 1, meta["num_layers"])
            del eig, vectors, mmap, calibration, factor
            gc.collect()
            torch.cuda.empty_cache()
    base.atomic_json(done, {
        "model": model_key, "model_id": meta["model_id"], "num_layers": meta["num_layers"],
        "hidden_size": meta["hidden_size"], "intermediate_size": meta["intermediate_size"],
        "group_size": GROUP_SIZE, "calibration_sequences": meta["sequences"],
        "sequence_length": meta["seq_len"], "permutation_token_stride": stride,
        "eigenspace": "frozen E1c randomized eigenspaces; no recomputation",
        "max_anchor_error": max(errors) if errors else 0.0,
    })


class SketchCollector:
    def __init__(self, model: torch.nn.Module, bases: dict[tuple[str, int], torch.Tensor], trace: bool):
        self.model = model
        self.bases = bases
        self.trace = trace
        self.results = {key: torch.zeros_like(value, dtype=torch.float64) for key, value in bases.items()}
        self.traces = {key: 0.0 for key in bases}
        self.counts = {key: 0 for key in bases}
        self.handles: list[Any] = []
        self.device_cache: dict[tuple[str, int], torch.Tensor] = {}

    def consume(self, site: str, layer: int, value: torch.Tensor) -> None:
        key = (site, layer)
        x = value.detach().float().reshape(-1, value.shape[-1])
        if key not in self.device_cache:
            self.device_cache[key] = self.bases[key].to(x.device, dtype=torch.float32)
        projected = x @ self.device_cache[key]
        self.results[key] += (x.T @ projected).double().cpu()
        if self.trace:
            self.traces[key] += float(x.square().sum(dtype=torch.float64).item())
        self.counts[key] += x.shape[0]

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
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.input_layernorm.register_forward_hook(self.q_hook(layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.down_hook(layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.device_cache.clear()


class EnergyCollector:
    def __init__(self, model: torch.nn.Module, vectors: dict[tuple[str, int], torch.Tensor], stride: int):
        self.model = model
        self.stride = stride
        self.reflectors: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, float]] = {}
        self.sums: dict[tuple[str, int], torch.Tensor] = {}
        self.counts: dict[tuple[str, int], int] = {}
        self.handles: list[Any] = []
        for key, value in vectors.items():
            refs, active, error = reflectors_from_vectors(value.to("cuda"), GROUP_SIZE)
            self.reflectors[key] = (refs, active, error)
            self.sums[key] = torch.zeros(value.shape[0], dtype=torch.float64)
            self.counts[key] = 0

    def consume(self, site: str, layer: int, value: torch.Tensor) -> None:
        key = (site, layer)
        sampled = value.detach()[:, :: self.stride, :].float().reshape(-1, value.shape[-1])
        refs, active, _ = self.reflectors[key]
        transformed = apply_reflectors(sampled, refs, active)
        self.sums[key] += transformed.square().sum(0).double().cpu()
        self.counts[key] += transformed.shape[0]

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
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.input_layernorm.register_forward_hook(self.q_hook(layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.down_hook(layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _model_pass(model: torch.nn.Module, tokens: torch.Tensor, batch_size: int, label: str) -> None:
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], batch_size):
            stop = min(start + batch_size, tokens.shape[0])
            model.model(input_ids=tokens[start:stop].cuda(non_blocking=True), use_cache=False)
            if stop % max(batch_size, 8) == 0 or stop == tokens.shape[0]:
                LOG.info("%s %d/%d sequences", label, stop, tokens.shape[0])


def calibrate_streamed_factors(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("factor calibration requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    model_id, model_key = model_id_and_key(args.model)
    base.setup_logging(workdir, f"e5-calibrate-{model_key}")
    if model_key == "llama32_3b" and not args.force_stream:
        build_factors_from_e1c(workdir, model_key, args.permutation_stride)
        return
    base.seed_everything(args.seed)
    output = factor_dir(workdir, model_key)
    done = output / "DONE.json"
    if done.exists():
        LOG.info("activation factor checkpoint exists: %s", done)
        return
    tokens = base.prepare_token_chunks(model_id, "train", 0, args.calibration_sequences, args.seq_len, workdir)
    model = base.load_model(model_id, workdir)
    layers = int(model.config.num_hidden_layers)
    dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
    bases: dict[tuple[str, int], torch.Tensor] = {}
    for site, n in dimensions.items():
        width = n // GROUP_SIZE + args.oversample
        for layer in range(layers):
            generator = torch.Generator(device="cpu").manual_seed(args.seed + 100_000 * (site == "down") + layer)
            omega = torch.randn((n, width), generator=generator, dtype=torch.float64)
            bases[(site, layer)] = torch.linalg.qr(omega, mode="reduced").Q.float()
    final_cq: dict[tuple[str, int], torch.Tensor] = {}
    traces: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    for pass_index in range(3):
        collector = SketchCollector(model, bases, trace=pass_index == 2)
        collector.install()
        try:
            _model_pass(model, tokens, args.batch_size, f"covariance pass {pass_index + 1}/3")
        finally:
            collector.close()
        for key, result in collector.results.items():
            normalized = result / collector.counts[key]
            if pass_index < 2:
                bases[key] = torch.linalg.qr(normalized, mode="reduced").Q.float()
            else:
                final_cq[key] = normalized
                traces[key] = collector.traces[key] / collector.counts[key]
                counts[key] = collector.counts[key]
        del collector
        gc.collect()
        torch.cuda.empty_cache()
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    eigenvalues: dict[tuple[str, int], torch.Tensor] = {}
    residuals: dict[tuple[str, int], torch.Tensor] = {}
    for key, q in bases.items():
        cq = final_cq[key]
        small = q.double().T @ cq.double()
        small = (small + small.T) / 2
        values, u = torch.linalg.eigh(small)
        rank = q.shape[0] // GROUP_SIZE
        order = torch.argsort(values, descending=True)[:rank]
        values = values[order].clamp_min(0)
        u = u[:, order]
        v = q.double() @ u
        residual = cq.double() @ u - v * values.unsqueeze(0)
        vectors[key] = v.float().cpu()
        eigenvalues[key] = values.cpu()
        residuals[key] = (residual.norm(dim=0) / values.clamp_min(torch.finfo(torch.float64).tiny)).cpu()
    energy = EnergyCollector(model, vectors, args.permutation_stride)
    energy.install()
    try:
        _model_pass(model, tokens, args.batch_size, "permutation-energy pass 1/1")
    finally:
        energy.close()
    output.mkdir(parents=True, exist_ok=True)
    eig_rows: list[dict[str, Any]] = []
    anchor_errors: list[float] = []
    for key, v in vectors.items():
        site, layer = key
        refs, active, error = energy.reflectors[key]
        means = energy.sums[key] / energy.counts[key]
        # balanced_orders only needs coordinate energies; synthesize one row whose squares match them.
        source, target = balanced_orders(means.clamp_min(0).sqrt().to("cuda").unsqueeze(0), v.shape[1], GROUP_SIZE)
        factor = RotationFactor(v.shape[0], GROUP_SIZE, refs, active, source.to("cuda"), target.to("cuda"), error)
        factor.save(output / f"{site}_layer_{layer:02d}.pt", {
            "source": "streamed fixed randomized eigenspace", "site": site, "layer": layer,
            "eigenvalues": eigenvalues[key], "trace": traces[key], "relative_ritz_residuals": residuals[key],
            "rows": counts[key],
        })
        anchor_errors.append(error)
        for index in range(v.shape[1]):
            eig_rows.append({
                "model": model_key, "site": site, "layer": layer, "rank": index + 1,
                "eigenvalue": float(eigenvalues[key][index]),
                "fraction_total_energy": float(eigenvalues[key][index]) / traces[key],
                "cumulative_fraction_total_energy": float(eigenvalues[key][: index + 1].sum()) / traces[key],
                "relative_ritz_residual": float(residuals[key][index]),
            })
    base.write_csv(workdir / "results" / model_key / "e5_calibration_eigenspace.csv", eig_rows)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "num_layers": layers,
        "hidden_size": dimensions["qkv"], "intermediate_size": dimensions["down"],
        "group_size": GROUP_SIZE, "calibration_split": "train", "calibration_offset": 0,
        "calibration_sequences": args.calibration_sequences, "sequence_length": args.seq_len,
        "subspace_oversample": args.oversample, "power_iterations": 1,
        "covariance_passes": 3, "permutation_energy_passes": 1,
        "permutation_token_stride": args.permutation_stride,
        "max_anchor_error": max(anchor_errors), "hardware": base.hardware_info(),
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()


class MethodRotations:
    def __init__(self, workdir: Path, model_key: str, method: str, seed_index: int,
                 base_seed: int, layers: int, dimensions: dict[str, int], device: torch.device):
        self.method = method
        self.factors: dict[tuple[str, int], RotationFactor] = {}
        self.signs: dict[tuple[str, int], torch.Tensor] = {}
        for site, n in dimensions.items():
            for layer in range(layers):
                generator = torch.Generator(device="cpu").manual_seed(_seed(base_seed, seed_index, layer, site))
                signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1)
                self.signs[(site, layer)] = signs.to(device)
                if method == "nar":
                    self.factors[(site, layer)] = RotationFactor.load(
                        factor_dir(workdir, model_key) / f"{site}_layer_{layer:02d}.pt", device
                    )

    def apply(self, site: str, layer: int, x: torch.Tensor) -> torch.Tensor:
        if self.method == "identity":
            return x.float()
        signs = self.signs[(site, layer)]
        if self.method == "hadamard":
            return full_hadamard_rows(x.float(), signs)
        if self.method == "nar":
            return self.factors[(site, layer)].apply(x, signs)
        raise ValueError(self.method)


class ActivationQuantHooks:
    def __init__(self, model: torch.nn.Module, rotations: MethodRotations,
                 qkv_enabled: bool, down_enabled: bool):
        self.model = model
        self.rotations = rotations
        self.qkv_enabled = qkv_enabled
        self.down_enabled = down_enabled
        self.handles: list[Any] = []

    def quantize(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        rotated = self.rotations.apply(site, layer, value)
        dequant, _, _, _ = base.dynamic_asym_int4(rotated, GROUP_SIZE)
        return dequant.to(value.dtype)

    def q_hook(self, layer: int) -> Callable[..., torch.Tensor]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            return self.quantize("qkv", layer, output)
        return hook

    def down_hook(self, layer: int) -> Callable[..., tuple[torch.Tensor]]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[torch.Tensor]:
            return (self.quantize("down", layer, inputs[0]),)
        return hook

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers):
            if self.qkv_enabled:
                self.handles.append(block.input_layernorm.register_forward_hook(self.q_hook(layer)))
            if self.down_enabled:
                self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.down_hook(layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class WeightManager:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.modules: dict[tuple[str, int], list[torch.nn.Linear]] = {}
        self.originals: dict[int, torch.Tensor] = {}
        for layer, block in enumerate(model.model.layers):
            qkv = [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]
            down = [block.mlp.down_proj]
            self.modules[("qkv", layer)] = qkv
            self.modules[("down", layer)] = down
            for module in qkv + down:
                self.originals[id(module)] = module.weight.detach().to("cpu", dtype=torch.bfloat16).clone()

    def restore(self, site: str) -> None:
        with torch.no_grad():
            for (candidate, _layer), modules in self.modules.items():
                if candidate != site:
                    continue
                for module in modules:
                    module.weight.copy_(self.originals[id(module)].to(module.weight.device))

    def rotate(self, site: str, rotations: MethodRotations, row_batch: int) -> float:
        if rotations.method == "identity":
            self.restore(site)
            return 0.0
        max_relative = 0.0
        with torch.no_grad():
            for (candidate, layer), modules in self.modules.items():
                if candidate != site:
                    continue
                for module in modules:
                    original = self.originals[id(module)]
                    transformed_chunks: list[torch.Tensor] = []
                    for start in range(0, original.shape[0], row_batch):
                        chunk = original[start : start + row_batch].to(module.weight.device).float()
                        transformed_chunks.append(rotations.apply(site, layer, chunk).to(torch.bfloat16))
                    transformed = torch.cat(transformed_chunks, 0)
                    module.weight.copy_(transformed)
                    # Algebraic fold audit on fixed rows; bf16 storage is the only non-exact component.
                    generator = torch.Generator(device="cpu").manual_seed(777 + layer + (site == "down") * 1000)
                    probe = torch.randn((2, original.shape[1]), generator=generator).to(module.weight.device)
                    rotated_probe = rotations.apply(site, layer, probe).float()
                    reference = probe @ original[: min(16, original.shape[0])].to(probe.device).float().T
                    folded = rotated_probe @ transformed[: min(16, transformed.shape[0])].float().T
                    relative = float((folded - reference).norm() / reference.norm().clamp_min(1e-30))
                    max_relative = max(max_relative, relative)
                    del transformed, transformed_chunks, probe, rotated_probe, reference, folded
        return max_relative


def evaluate_nlls(model: torch.nn.Module, tokens: torch.Tensor, label: str) -> list[float]:
    losses: list[float] = []
    with torch.inference_mode():
        for index in range(tokens.shape[0]):
            batch = tokens[index : index + 1].cuda(non_blocking=True)
            output = model(input_ids=batch, labels=batch, use_cache=False)
            loss = float(output.loss.detach().float().cpu())
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite {label} loss at sequence {index}")
            losses.append(loss)
            if index % 8 == 0:
                LOG.info("%s sequence %d/%d loss=%.6f", label, index + 1, tokens.shape[0], loss)
    return losses


def _condition_flags(condition: str) -> tuple[bool, bool]:
    return condition in ("qkv_only", "both"), condition in ("down_only", "both")


def summarize_e5(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    model = str(rows[0]["model"])
    reference_rows = [r for r in rows if r["method"] == "bf16"]
    reference_ppl = math.exp(float(np.mean([float(r["nll"]) for r in reference_rows])))
    result.append({
        "model": model, "site": "none", "method": "bf16", "seeds": 1,
        "mean_ppl": reference_ppl, "seed_ppl_std": 0.0,
        "paired_ppl_delta_vs_hadamard": math.nan, "paired_90ci_low_vs_hadamard": math.nan,
        "paired_90ci_high_vs_hadamard": math.nan, "paired_ppl_delta_vs_identity": math.nan,
        "paired_90ci_low_vs_identity": math.nan, "paired_90ci_high_vs_identity": math.nan,
        "ppl_delta_vs_bf16": 0.0, "seed_ppls": f"{reference_ppl:.9g}",
    })
    for site in EVAL_SITES:
        site_rows = [r for r in rows if r["site"] == site and r["method"] != "bf16"]
        seeds = sorted({int(r["seed"]) for r in site_rows})
        ppls: dict[tuple[int, str], float] = {}
        for seed in seeds:
            for method in METHODS:
                values = [float(r["nll"]) for r in site_rows if int(r["seed"]) == seed and r["method"] == method]
                ppls[(seed, method)] = math.exp(float(np.mean(values)))
        for method in METHODS:
            values = np.asarray([ppls[(seed, method)] for seed in seeds])
            had_delta = np.asarray([ppls[(seed, method)] - ppls[(seed, "hadamard")] for seed in seeds])
            id_delta = np.asarray([ppls[(seed, method)] - ppls[(seed, "identity")] for seed in seeds])
            had_half = TCRIT_DF2_90 * float(had_delta.std(ddof=1)) / math.sqrt(len(seeds))
            id_half = TCRIT_DF2_90 * float(id_delta.std(ddof=1)) / math.sqrt(len(seeds))
            result.append({
                "model": model, "site": site, "method": method, "seeds": len(seeds),
                "mean_ppl": float(values.mean()), "seed_ppl_std": float(values.std(ddof=1)),
                "paired_ppl_delta_vs_hadamard": float(had_delta.mean()),
                "paired_90ci_low_vs_hadamard": float(had_delta.mean() - had_half),
                "paired_90ci_high_vs_hadamard": float(had_delta.mean() + had_half),
                "paired_ppl_delta_vs_identity": float(id_delta.mean()),
                "paired_90ci_low_vs_identity": float(id_delta.mean() - id_half),
                "paired_90ci_high_vs_identity": float(id_delta.mean() + id_half),
                "ppl_delta_vs_bf16": float(values.mean() - reference_ppl),
                "seed_ppls": ";".join(f"{value:.9g}" for value in values),
            })
    return result


def run_e5(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E5 requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    model_id, model_key = model_id_and_key(args.model)
    base.setup_logging(workdir, f"e5-ppl-{model_key}")
    done = workdir / "results" / model_key / "E5_DONE.json"
    if done.exists():
        LOG.info("E5 checkpoint exists: %s", done)
        return
    factor_meta = _json(factor_dir(workdir, model_key) / "DONE.json")
    tokens = base.prepare_token_chunks(model_id, "test", 0, args.eval_sequences, args.seq_len, workdir)
    model = base.load_model(model_id, workdir)
    layers = int(model.config.num_hidden_layers)
    dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
    if factor_meta["hidden_size"] != dimensions["qkv"] or factor_meta["intermediate_size"] != dimensions["down"]:
        raise RuntimeError("factor/model dimensions disagree")
    output = workdir / "results" / model_key / "e5_per_sequence.partial.csv"
    rows: list[dict[str, Any]] = list(_rows(output))
    completed = {(str(r["site"]), str(r["method"]), int(r["seed"])) for r in rows}
    if not any(r["method"] == "bf16" for r in rows):
        losses = evaluate_nlls(model, tokens, f"{model_key} bf16")
        rows.extend({
            "model": model_key, "model_id": model_id, "site": "none", "method": "bf16", "seed": -1,
            "sequence": index, "nll": loss, "tokens_scored": args.seq_len - 1,
            "reused_deterministic_run": False, "weight_fold_max_relative_error": 0.0,
        } for index, loss in enumerate(losses))
        base.write_csv(output, rows)
    weights = WeightManager(model)
    identity_cache: dict[str, list[float]] = {}
    for condition in EVAL_SITES:
        cached = sorted(
            (r for r in rows if r["site"] == condition and r["method"] == "identity" and int(r["seed"]) == args.seed),
            key=lambda r: int(r["sequence"]),
        )
        if cached:
            identity_cache[condition] = [float(r["nll"]) for r in cached]
    audit_partial = workdir / "results" / model_key / "e5_weight_fold_audit.partial.csv"
    fold_audit: list[dict[str, Any]] = list(_rows(audit_partial))
    for method in METHODS:
        seed_indices = range(args.seeds) if method != "identity" else range(1)
        for seed_index in seed_indices:
            seed_value = args.seed + seed_index
            rotations = MethodRotations(workdir, model_key, method, seed_index, args.seed, layers, dimensions, torch.device("cuda"))
            weights.restore("qkv")
            weights.restore("down")
            q_error = weights.rotate("qkv", rotations, args.weight_row_batch)
            fold_audit.append({"method": method, "seed": seed_value, "site": "qkv", "max_relative_error": q_error})
            base.write_csv(audit_partial, fold_audit)
            for condition in ("qkv_only", "both", "down_only"):
                if condition == "both":
                    d_error = weights.rotate("down", rotations, args.weight_row_batch)
                    fold_audit.append({"method": method, "seed": seed_value, "site": "down", "max_relative_error": d_error})
                    base.write_csv(audit_partial, fold_audit)
                elif condition == "down_only":
                    weights.restore("qkv")
                key_seed = args.seed if method == "identity" else seed_value
                key = (condition, method, key_seed)
                if key in completed:
                    continue
                q_enabled, d_enabled = _condition_flags(condition)
                hooks = ActivationQuantHooks(model, rotations, q_enabled, d_enabled)
                hooks.install()
                try:
                    losses = evaluate_nlls(model, tokens, f"{model_key} {condition} {method} seed={seed_value}")
                finally:
                    hooks.close()
                if method == "identity":
                    identity_cache[condition] = losses
                rows.extend({
                    "model": model_key, "model_id": model_id, "site": condition, "method": method,
                    "seed": key_seed, "sequence": index, "nll": loss,
                    "tokens_scored": args.seq_len - 1, "reused_deterministic_run": False,
                    "weight_fold_max_relative_error": max(q_error, d_error if condition in ("both", "down_only") else 0.0),
                } for index, loss in enumerate(losses))
                base.write_csv(output, rows)
            weights.restore("qkv")
            weights.restore("down")
            del rotations
            gc.collect()
            torch.cuda.empty_cache()
    # Replicate deterministic identity rows across seeds for paired summaries.
    existing = {(str(r["site"]), str(r["method"]), int(r["seed"])) for r in rows}
    for condition, losses in identity_cache.items():
        for seed_index in range(1, args.seeds):
            seed_value = args.seed + seed_index
            if (condition, "identity", seed_value) in existing:
                continue
            rows.extend({
                "model": model_key, "model_id": model_id, "site": condition, "method": "identity",
                "seed": seed_value, "sequence": index, "nll": loss,
                "tokens_scored": args.seq_len - 1, "reused_deterministic_run": True,
                "weight_fold_max_relative_error": 0.0,
            } for index, loss in enumerate(losses))
    final_raw = workdir / "results" / model_key / "e5_per_sequence.csv"
    base.write_csv(final_raw, rows)
    base.write_csv(workdir / "results" / model_key / "e5_summary.csv", summarize_e5(rows))
    base.write_csv(workdir / "results" / model_key / "e5_weight_fold_audit.csv", fold_audit)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "group_size": GROUP_SIZE, "bits": 4,
        "scale_dtype": "fp16", "zero_point_dtype": "fp16 real-valued offset",
        "eval_split": "test", "eval_sequences": args.eval_sequences, "sequence_length": args.seq_len,
        "seeds": args.seeds, "base_seed": args.seed,
        "sites": list(EVAL_SITES), "methods": list(METHODS),
        "paired_design": "identical WikiText-2 sequences within each seed/site; only activation rotation differs",
        "ci": "two-sided paired 90% Student-t CI over three seed-level PPL differences (df=2)",
        "weights": "bf16; R^T is folded into q/k/v or down_proj weight rows offline",
        "other_tensors": "KV and all non-target activations remain bf16",
        "calibration": factor_meta, "hardware": base.hardware_info(),
    })
    output.unlink(missing_ok=True)
    audit_partial.unlink(missing_ok=True)
    del model, weights
    gc.collect()
    torch.cuda.empty_cache()


def run_e6(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E6 requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    model_id, model_key = model_id_and_key(args.model)
    if model_key != "llama32_3b":
        raise ValueError("E6 is frozen to Llama-3.2-3B down_proj")
    base.setup_logging(workdir, "e6-online-cost")
    result_dir = workdir / "results" / model_key
    done = result_dir / "E6_DONE.json"
    if done.exists():
        LOG.info("E6 checkpoint exists: %s", done)
        return
    meta = _json(factor_dir(workdir, model_key) / "DONE.json")
    n = int(meta["intermediate_size"])
    hidden = int(meta["hidden_size"])
    factor = RotationFactor.load(factor_dir(workdir, model_key) / "down_layer_00.pt", torch.device("cuda"))
    rotations = MethodRotations(workdir, model_key, "nar", 0, args.seed, int(meta["num_layers"]),
                                {"qkv": hidden, "down": n}, torch.device("cuda"))
    signs = rotations.signs[("down", 0)]
    # Materialize R^T once only as a verification reference, never as the implementation.
    dense_chunks: list[torch.Tensor] = []
    eye = torch.eye(n, device="cuda", dtype=torch.float32)
    for start in range(0, n, args.dense_row_batch):
        dense_chunks.append(factor.apply(eye[start : start + args.dense_row_batch], signs))
    dense_rt = torch.cat(dense_chunks, 0)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    probe = torch.randn((args.verify_rows, n), generator=generator).cuda()
    factored = factor.apply(probe, signs)
    dense = probe @ dense_rt
    diff = factored - dense
    verify = {
        "max_abs_error": float(diff.abs().max()),
        "relative_l2_error": float(diff.norm() / dense.norm()),
        "rows": args.verify_rows,
        "layer": 0,
    }
    del eye, dense_chunks, dense_rt, probe, factored, dense, diff
    torch.cuda.empty_cache()

    reflectors = int(factor.active.sum())
    flop_rows: list[dict[str, Any]] = []
    down_flops = 2 * n * hidden
    nar_flops = 4 * n * reflectors + n * int(math.log2(GROUP_SIZE)) + n
    had_flops = n * int(math.log2(n)) + n
    weight = torch.randn((hidden, n), device="cuda", dtype=torch.bfloat16) / math.sqrt(n)

    def benchmark(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(stop) / repeats)

    for tokens in args.benchmark_tokens:
        x = torch.randn((tokens, n), device="cuda", dtype=torch.float32)
        x_bf16 = x.to(torch.bfloat16)
        nar_ms = benchmark(lambda: factor.apply(x, signs), args.warmup, args.repeats)
        had_ms = benchmark(lambda: full_hadamard_rows(x, signs), args.warmup, args.repeats)
        matmul_ms = benchmark(lambda: torch.nn.functional.linear(x_bf16, weight), args.warmup, args.repeats)
        flop_rows.append({
            "model": model_key, "layer": 0, "tokens": tokens, "n": n, "hidden": hidden,
            "k": n // GROUP_SIZE, "householder_reflections": reflectors,
            "nar_flops_per_token": nar_flops, "hadamard_flops_per_token": had_flops,
            "down_matmul_flops_per_token": down_flops,
            "nar_flop_ratio_vs_matmul": nar_flops / down_flops,
            "hadamard_flop_ratio_vs_matmul": had_flops / down_flops,
            "nar_ms": nar_ms, "hadamard_ms": had_ms, "down_matmul_ms": matmul_ms,
            "nar_wall_ratio_vs_hadamard": nar_ms / had_ms,
            "nar_wall_ratio_vs_down_matmul": nar_ms / matmul_ms,
            "hadamard_wall_ratio_vs_down_matmul": had_ms / matmul_ms,
            "nar_exceeds_10pct_matmul_wall": nar_ms / matmul_ms > 0.10,
        })
        del x, x_bf16
    base.write_csv(result_dir / "e6_online_cost.csv", flop_rows)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "site": "down_proj input",
        "factorization": "k sequential Householder reflections, fixed permutation, sign, block H128",
        "maximum_allowed_reflections": 2 * (n // GROUP_SIZE),
        "actual_reflections": reflectors, "dense_verification": verify,
        "benchmark": {"warmup": args.warmup, "repeats": args.repeats, "token_counts": args.benchmark_tokens},
        "hardware": base.hardware_info(),
    })


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    calibrate.add_argument("--calibration-sequences", type=int, default=128)
    calibrate.add_argument("--seq-len", type=int, default=2048)
    calibrate.add_argument("--batch-size", type=int, default=1)
    calibrate.add_argument("--oversample", type=int, default=16)
    calibrate.add_argument("--permutation-stride", type=int, default=32)
    calibrate.add_argument("--force-stream", action="store_true")
    e5 = sub.add_parser("e5")
    e5.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    e5.add_argument("--eval-sequences", type=int, default=64)
    e5.add_argument("--seq-len", type=int, default=2048)
    e5.add_argument("--seeds", type=int, default=3)
    e5.add_argument("--weight-row-batch", type=int, default=512)
    e6 = sub.add_parser("e6")
    e6.add_argument("--model", default="llama32_3b", choices=tuple(MODEL_IDS))
    e6.add_argument("--verify-rows", type=int, default=8)
    e6.add_argument("--dense-row-batch", type=int, default=256)
    e6.add_argument("--benchmark-tokens", type=int, nargs="+", default=[1, 32, 2048])
    e6.add_argument("--warmup", type=int, default=10)
    e6.add_argument("--repeats", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    {"calibrate": calibrate_streamed_factors, "e5": run_e5, "e6": run_e6}[args.command](args)


if __name__ == "__main__":
    main()
