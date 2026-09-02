#!/usr/bin/env python3
"""E14 end-to-end W4A4KV4 evaluation built on pinned QuaRot GPTQ.

Rows are frozen before execution.  The QuaRot-derived numerical choices and
the deliberate deviations requested by the experiment contract are recorded
in every completion manifest.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from .e12_wy import WYFactor, compact_wy
    from .quarot_gptq import GPTQ
except ImportError:
    import activation_experiments as act
    import experiment as base
    from e12_wy import WYFactor, compact_wy
    from quarot_gptq import GPTQ


LOG = logging.getLogger("nar")
MODELS = ("llama32_3b", "llama31_8b")
ROTATIONS = ("hadamard", "nar")
ROWS = ("quarot", "hadamard_asym_g128", "nar_asym_g128")
TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "lambada_openai")
METRICS = {
    "piqa": "acc_norm,none", "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none", "hellaswag": "acc_norm,none",
    "winogrande": "acc,none", "lambada_openai": "acc,none",
}
QUAROT_COMMIT = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
HARNESS_COMMIT = "b954108c9baaaa934b4ad842033b31a97ee30816"
GROUP = 128
K_TOKEN_GROUP = 32


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rotation_dir(workdir: Path, model: str) -> Path:
    return workdir / "activations" / model / "e14_rotations"


def checkpoint_dir(artifact_root: Path, model: str, rotation: str) -> Path:
    return artifact_root / model / f"gptq_{rotation}"


def _seed(base_seed: int, label: str, layer: int = 0) -> int:
    offsets = {"r1": 140_000, "r2": 240_000, "r4": 340_000}
    return base_seed + offsets[label] + layer


def _signs(n: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=generator).float().mul_(2).sub_(1).to(device)


def _balanced_orders_from_energy(energy: torch.Tensor, rank: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = energy.numel()
    groups = n // b
    anchors = [index * b for index in range(rank)]
    remaining = [index for index in range(n) if index not in anchors]
    fillers = sorted(remaining, key=lambda index: (float(energy[index]), index))[: groups - rank]
    source = anchors + fillers
    residual = [index for index in remaining if index not in fillers]
    residual.sort(key=lambda index: (-float(energy[index]), index))
    target = [group * b for group in range(groups)] + base._balanced_target_slots(
        [max(0.0, float(energy[index])) for index in residual], groups, b
    )
    source.extend(residual)
    return torch.tensor(source), torch.tensor(target)


class RotationCalibrationCollector:
    """Shared R1 sketch plus exact per-layer V second moments."""

    def __init__(self, model: torch.nn.Module, basis: torch.Tensor, collect_v: bool):
        self.model = model
        self.basis = basis
        self.collect_v = collect_v
        self.projected = torch.zeros_like(basis, dtype=torch.float64)
        self.trace = 0.0
        self.count = 0
        head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
        self.head_dim = head_dim
        self.v_moments = [torch.zeros((head_dim, head_dim), dtype=torch.float64) for _ in model.model.layers]
        self.v_counts = [0 for _ in model.model.layers]
        self.handles: list[Any] = []
        self.device_basis: torch.Tensor | None = None

    def consume_r1(self, value: torch.Tensor) -> None:
        rows = value.detach().float().reshape(-1, value.shape[-1])
        if self.device_basis is None:
            self.device_basis = self.basis.to(rows.device)
        projection = rows @ self.device_basis
        self.projected += (rows.T @ projection).double().cpu()
        self.trace += float(rows.square().sum(dtype=torch.float64))
        self.count += rows.shape[0]

    def r1_hook(self, _layer: int) -> Callable[..., torch.Tensor]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            self.consume_r1(output)
            return output
        return hook

    def v_hook(self, layer: int) -> Callable[..., torch.Tensor]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            if self.collect_v:
                rows = output.detach().float().reshape(-1, self.head_dim)
                self.v_moments[layer] += (rows.T @ rows).double().cpu()
                self.v_counts[layer] += rows.shape[0]
            return output
        return hook

    def install(self) -> None:
        for index, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.input_layernorm.register_forward_hook(self.r1_hook(index)))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self.v_hook(index)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.device_basis = None


class RotationEnergyCollector:
    def __init__(self, model: torch.nn.Module, reflectors: torch.Tensor, active: torch.Tensor):
        self.model = model
        self.reflectors = reflectors
        self.active = active
        self.energy = torch.zeros(reflectors.shape[1], dtype=torch.float64)
        self.count = 0
        self.handles: list[Any] = []

    def hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        rows = output.detach()[:, ::32, :].float().reshape(-1, output.shape[-1])
        transformed = act.apply_reflectors(rows, self.reflectors, self.active)
        self.energy += transformed.square().sum(0).double().cpu()
        self.count += transformed.shape[0]
        return output

    def install(self) -> None:
        for layer in self.model.model.layers:
            self.handles.append(layer.input_layernorm.register_forward_hook(self.hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _model_pass(model: torch.nn.Module, tokens: torch.Tensor, label: str) -> None:
    with torch.inference_mode():
        for index in range(tokens.shape[0]):
            model.model(input_ids=tokens[index:index + 1].cuda(), use_cache=False)
            if index % 16 == 0:
                LOG.info("%s %d/%d", label, index + 1, tokens.shape[0])


def calibrate_rotations(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E14 rotation calibration requires CUDA")
    workdir = Path(args.workdir).resolve()
    output = rotation_dir(workdir, args.model)
    done = output / "DONE.json"
    if done.exists():
        LOG.info("E14 rotation calibration exists: %s", done)
        return
    base.setup_logging(workdir, f"e14-calibrate-{args.model}")
    base.seed_everything(args.seed)
    model_id, model_key = act.model_id_and_key(args.model)
    tokens = base.prepare_token_chunks(model_id, "train", 0, args.calibration_sequences, args.seq_len, workdir)
    model = base.load_model(model_id, workdir)
    n = int(model.config.hidden_size)
    rank = n // GROUP
    generator = torch.Generator(device="cpu").manual_seed(_seed(args.seed, "r1"))
    basis = torch.linalg.qr(torch.randn((n, rank + 16), generator=generator, dtype=torch.float64), mode="reduced").Q.float()
    final_projected = None
    trace = 0.0
    count = 0
    v_moments: list[torch.Tensor] = []
    v_counts: list[int] = []
    for pass_index in range(3):
        collector = RotationCalibrationCollector(model, basis, collect_v=pass_index == 2)
        collector.install()
        try:
            _model_pass(model, tokens, f"E14 covariance pass {pass_index + 1}/3")
        finally:
            collector.close()
        normalized = collector.projected / collector.count
        if pass_index < 2:
            basis = torch.linalg.qr(normalized, mode="reduced").Q.float()
        else:
            final_projected = normalized
            trace = collector.trace / collector.count
            count = collector.count
            v_moments = collector.v_moments
            v_counts = collector.v_counts
        del collector
        gc.collect()
        torch.cuda.empty_cache()
    assert final_projected is not None
    small = basis.double().T @ final_projected.double()
    values, vectors_small = torch.linalg.eigh((small + small.T) / 2)
    order = torch.argsort(values, descending=True)[:rank]
    values = values[order].clamp_min(0)
    vectors = (basis.double() @ vectors_small[:, order]).float().cuda()
    cq_vectors = final_projected.double() @ vectors_small[:, order]
    residuals = (cq_vectors - vectors.double().cpu() * values.unsqueeze(0)).norm(dim=0) / values.clamp_min(1e-30)
    reflectors, active, anchor_error = act.reflectors_from_vectors(vectors, GROUP)
    energy_collector = RotationEnergyCollector(model, reflectors, active)
    energy_collector.install()
    try:
        _model_pass(model, tokens, "E14 R1 permutation energy")
    finally:
        energy_collector.close()
    source, target = _balanced_orders_from_energy(energy_collector.energy / energy_collector.count, rank, GROUP)
    output.mkdir(parents=True, exist_ok=True)
    r1 = act.RotationFactor(n, GROUP, reflectors, active, source.cuda(), target.cuda(), anchor_error)
    r1.save(output / "r1.pt", {
        "eigenvalues": values, "trace": trace, "relative_ritz_residuals": residuals,
        "rows": count, "definition": "top directions of post-input-RMSNorm activations pooled equally by token across all layers",
    })
    v_rows = []
    for layer, moment in enumerate(v_moments):
        covariance = moment / v_counts[layer]
        v_values, v_vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
        vector = v_vectors[:, -1:].float().cuda()
        refs, enabled, error = act.reflectors_from_vectors(vector, GROUP)
        identity = torch.arange(vector.shape[0], device="cuda")
        factor = act.RotationFactor(vector.shape[0], GROUP, refs, enabled, identity, identity, error)
        factor.save(output / f"r2_v_layer_{layer:02d}.pt", {
            "top_eigenvalue": float(v_values[-1]), "trace": float(v_values.clamp_min(0).sum()),
            "rows": v_counts[layer], "definition": "top per-head V second-moment direction pooled across KV heads",
        })
        v_rows.append({"model": model_key, "layer": layer, "top_eigenvalue": float(v_values[-1]),
                       "trace": float(v_values.clamp_min(0).sum()), "rows": v_counts[layer],
                       "anchor_error": error})
    base.write_csv(workdir / "results" / model_key / "e14_rotation_calibration.csv", v_rows)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "calibration_sequences": args.calibration_sequences,
        "sequence_length": args.seq_len, "covariance_passes": 3,
        "r1": "single global pooled post-RMSNorm NAR, group-128, k=n/128",
        "r2": "per-layer, per-head V NAR, head_dim=128, group-128, k=1",
        "r4": "reuse frozen E5 per-layer down-input NAR factors",
        "seed": args.seed, "hardware": base.hardware_info(),
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()


class RotationSet:
    def __init__(self, workdir: Path, model_key: str, method: str, seed: int,
                 config: Any, device: torch.device):
        self.method = method
        self.seed = seed
        self.device = device
        self.layers = int(config.num_hidden_layers)
        self.hidden = int(config.hidden_size)
        self.intermediate = int(config.intermediate_size)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.r1 = None
        self.r2: dict[int, act.RotationFactor] = {}
        self.r4: dict[int, act.RotationFactor] = {}
        self.r4_wy: dict[int, WYFactor] = {}
        if method == "nar":
            root = rotation_dir(workdir, model_key)
            if not (root / "DONE.json").exists():
                raise FileNotFoundError(root / "DONE.json")
            self.r1 = act.RotationFactor.load(root / "r1.pt", device)
            for layer in range(self.layers):
                self.r2[layer] = act.RotationFactor.load(root / f"r2_v_layer_{layer:02d}.pt", device)
                self.r4[layer] = act.RotationFactor.load(
                    act.factor_dir(workdir, model_key) / f"down_layer_{layer:02d}.pt", device
                )
                w, y = compact_wy(self.r4[layer].reflectors, self.r4[layer].active)
                self.r4_wy[layer] = WYFactor(self.r4[layer], w, y)
        self.sign_cache: dict[tuple[str, int], torch.Tensor] = {}

    def signs(self, label: str, layer: int, n: int) -> torch.Tensor:
        key = (label, layer)
        if key not in self.sign_cache:
            self.sign_cache[key] = _signs(n, _seed(self.seed, label, layer), self.device)
        return self.sign_cache[key]

    def apply(self, label: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        signs = self.signs(label, layer, value.shape[-1])
        if self.method == "hadamard":
            if label != "r1":
                signs = torch.ones_like(signs)
            return act.full_hadamard_rows(value.float(), signs)
        if label == "r1":
            assert self.r1 is not None
            return self.r1.apply(value, signs)
        if label == "r2":
            return self.r2[layer].apply(value, signs)
        return self.r4_wy[layer].apply(value, signs)


def _transform_weight_rows(module: torch.nn.Linear, transform: Callable[[torch.Tensor], torch.Tensor],
                           row_batch: int) -> None:
    original = module.weight.detach()
    chunks = []
    for start in range(0, original.shape[0], row_batch):
        chunks.append(transform(original[start:start + row_batch].float()).to(original.dtype))
    module.weight.data.copy_(torch.cat(chunks, 0))


def _transform_weight_left(module: torch.nn.Linear, transform: Callable[[torch.Tensor], torch.Tensor],
                           row_batch: int) -> None:
    transposed = module.weight.detach().T
    chunks = []
    for start in range(0, transposed.shape[0], row_batch):
        chunks.append(transform(transposed[start:start + row_batch].float()).to(transposed.dtype))
    module.weight.data.copy_(torch.cat(chunks, 0).T)


@torch.inference_mode()
def fuse_norms_and_rotate(model: torch.nn.Module, rotations: RotationSet,
                          row_batch: int) -> dict[str, Any]:
    """Apply QuaRot R1 plus R2/R4 folds with explicit orthogonal identities."""
    tied_embeddings = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    if tied_embeddings:
        old_head = model.lm_head
        new_head = torch.nn.Linear(old_head.in_features, old_head.out_features, bias=False,
                                   device=old_head.weight.device, dtype=old_head.weight.dtype)
        new_head.weight.copy_(old_head.weight)
        model.lm_head = new_head
        model.config.tie_word_embeddings = False
    for block in model.model.layers:
        input_scale = block.input_layernorm.weight.detach().float()
        post_scale = block.post_attention_layernorm.weight.detach().float()
        for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj):
            module.weight.mul_(input_scale.to(module.weight.dtype).unsqueeze(0))
        for module in (block.mlp.gate_proj, block.mlp.up_proj):
            module.weight.mul_(post_scale.to(module.weight.dtype).unsqueeze(0))
        block.input_layernorm.weight.fill_(1)
        block.post_attention_layernorm.weight.fill_(1)
    final_scale = model.model.norm.weight.detach().float()
    model.lm_head.weight.mul_(final_scale.to(model.lm_head.weight.dtype).unsqueeze(0))
    model.model.norm.weight.fill_(1)

    r1 = lambda value: rotations.apply("r1", 0, value)
    _transform_weight_rows(model.model.embed_tokens, r1, row_batch)
    _transform_weight_rows(model.lm_head, r1, row_batch)
    for block in model.model.layers:
        for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj,
                       block.mlp.gate_proj, block.mlp.up_proj):
            _transform_weight_rows(module, r1, row_batch)
        for module in (block.self_attn.o_proj, block.mlp.down_proj):
            _transform_weight_left(module, r1, row_batch)

    attention_heads = int(model.config.num_attention_heads)
    head_dim = rotations.head_dim
    for layer, block in enumerate(model.model.layers):
        _transform_weight_rows(
            block.mlp.down_proj, lambda value, layer=layer: rotations.apply("r4", layer, value), row_batch
        )
        weight = block.self_attn.o_proj.weight.detach()
        shaped = weight.reshape(weight.shape[0], attention_heads, head_dim)
        rotated = rotations.apply("r2", layer, shaped.reshape(-1, head_dim)).to(weight.dtype)
        weight.copy_(rotated.reshape_as(shaped).reshape_as(weight))
    return {
        "r1": "global residual rotation; input rows WQ, residual output Q^T W",
        "r2": "per-head V rotation with identical fold into every GQA-expanded o_proj head block",
        "r4": "per-layer down-input rotation folded into down_proj rows",
        "norm": "all RMSNorm affine weights fused into consumer weights then set to one",
        "embedding_centering": False,
        "tied_embeddings_materialized_before_final_norm_fusion": tied_embeddings,
        "official_bug_avoided": "official head calls rotate_embeddings twice; E14 applies algebraic R1 once",
    }


def _symmetric_per_token_int4(value: torch.Tensor) -> torch.Tensor:
    original_dtype = value.dtype
    rows = value.float().reshape(-1, value.shape[-1])
    scale = (rows.abs().amax(-1, keepdim=True) / 7).clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
    dequant = torch.round(rows / scale.float()).clamp_(-8, 7) * scale.float()
    return dequant.reshape_as(value).to(original_dtype)


def _kivi_key_qdq(key: torch.Tensor) -> torch.Tensor:
    """Asymmetric per-channel K, contiguous token groups; short tail is bf16."""
    length = key.shape[-2]
    prefix = length // K_TOKEN_GROUP * K_TOKEN_GROUP
    if prefix == 0:
        return key
    transposed = key[..., :prefix, :].transpose(-1, -2).contiguous()
    quantized, _, _, _ = base.dynamic_asym_int4(transposed, K_TOKEN_GROUP)
    output = key.clone()
    output[..., :prefix, :] = quantized.transpose(-1, -2)
    return output


class RuntimeHooks:
    def __init__(self, model: torch.nn.Module, rotations: RotationSet,
                 activation_kind: str | None, quantize_kv: bool = True):
        self.model = model
        self.rotations = rotations
        self.activation_kind = activation_kind
        self.quantize_kv = quantize_kv
        self.handles: list[Any] = []
        self.previous_attention = model.config._attn_implementation
        self.attention_key = f"nar_e14_{id(self)}"

    def rotate_down(self, layer: int) -> Callable[..., tuple[torch.Tensor]]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[torch.Tensor]:
            return (self.rotations.apply("r4", layer, inputs[0]).to(inputs[0].dtype),)
        return hook

    def rotate_v(self, layer: int) -> Callable[..., torch.Tensor]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            shape = output.shape
            return self.rotations.apply("r2", layer, output.reshape(-1, self.rotations.head_dim)).reshape(shape).to(output.dtype)
        return hook

    def quantize_input(self, _module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
        value = inputs[0]
        if self.activation_kind == "quarot_symmetric_token":
            quantized = _symmetric_per_token_int4(value)
        elif self.activation_kind == "asymmetric_g128":
            quantized, _, _, _ = base.dynamic_asym_int4(value, GROUP)
        else:
            return inputs
        return (quantized.to(value.dtype),) + inputs[1:]

    def attention(self, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, attention_mask: torch.Tensor | None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        quantized_key = _kivi_key_qdq(key)
        quantized_value, _, _, _ = base.dynamic_asym_int4(value, self.rotations.head_dim)
        return sdpa_attention_forward(module, query, quantized_key, quantized_value, attention_mask, **kwargs)

    def install(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if self.quantize_kv:
            ALL_ATTENTION_FUNCTIONS.register(self.attention_key, self.attention)
            self.model.config._attn_implementation = self.attention_key
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.self_attn.v_proj.register_forward_hook(self.rotate_v(layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self.rotate_down(layer)))
            if self.activation_kind is not None:
                for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj,
                               block.self_attn.o_proj, block.mlp.gate_proj, block.mlp.up_proj,
                               block.mlp.down_proj):
                    self.handles.append(module.register_forward_pre_hook(self.quantize_input))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model.config._attn_implementation = self.previous_attention


def _linear_groups(layer: torch.nn.Module) -> list[list[tuple[str, torch.nn.Linear]]]:
    return [
        [("self_attn.k_proj", layer.self_attn.k_proj), ("self_attn.v_proj", layer.self_attn.v_proj),
         ("self_attn.q_proj", layer.self_attn.q_proj)],
        [("self_attn.o_proj", layer.self_attn.o_proj)],
        [("mlp.up_proj", layer.mlp.up_proj), ("mlp.gate_proj", layer.mlp.gate_proj)],
        [("mlp.down_proj", layer.mlp.down_proj)],
    ]


def _layer_state(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().cpu() for group in _linear_groups(layer) for name, module in group}


def _load_layer_state(layer: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    modules = {name: module for group in _linear_groups(layer) for name, module in group}
    if set(payload) != set(modules):
        raise RuntimeError(f"checkpoint schema mismatch: {path}")
    for name, module in modules.items():
        module.weight.data.copy_(payload[name].to(module.weight.dtype))


def _layer_forward(layer: torch.nn.Module, hidden: torch.Tensor, position_ids: torch.Tensor,
                   position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    output = layer(
        hidden, attention_mask=None, position_ids=position_ids, use_cache=False,
        position_embeddings=position_embeddings,
    )
    return output[0] if isinstance(output, tuple) else output


def _prepare_rotated_model(workdir: Path, model_key: str, rotation: str,
                           seed: int, row_batch: int) -> tuple[torch.nn.Module, RotationSet, dict[str, Any]]:
    model_id, canonical = act.model_id_and_key(model_key)
    model = base.load_model(model_id, workdir)
    rotations = RotationSet(workdir, canonical, rotation, seed, model.config, torch.device("cuda"))
    fold = fuse_norms_and_rotate(model, rotations, row_batch)
    return model, rotations, fold


@torch.inference_mode()
def _fold_invariance(model_id: str, model: torch.nn.Module, rotations: RotationSet,
                     probe: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    hooks = RuntimeHooks(model, rotations, activation_kind=None, quantize_kv=False)
    hooks.install()
    try:
        observed = model(input_ids=probe, use_cache=False).logits.float()
    finally:
        hooks.close()
    difference = observed - reference
    return {
        "max_abs_logit_error": float(difference.abs().max()),
        "relative_l2_logit_error": float(difference.norm() / reference.norm().clamp_min(1e-30)),
        "probe_tokens": int(probe.numel()),
    }


def verify_rotation(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E14 verification requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, f"e14-verify-{args.model}-{args.rotation}")
    model_id, model_key = act.model_id_and_key(args.model)
    tokens = base.prepare_token_chunks(model_id, "train", 0, 1, args.seq_len, workdir)
    probe = tokens[:, :args.verify_tokens].cuda()
    original = base.load_model(model_id, workdir)
    with torch.inference_mode():
        reference = original(input_ids=probe, use_cache=False).logits.float()
    del original
    gc.collect()
    torch.cuda.empty_cache()
    model, rotations, fold = _prepare_rotated_model(
        workdir, model_key, args.rotation, args.seed, args.weight_row_batch
    )
    audit = _fold_invariance(model_id, model, rotations, probe, reference)
    if audit["relative_l2_logit_error"] > args.fold_tolerance:
        raise AssertionError(f"rotation fold failed tolerance: {audit}")
    base.atomic_json(workdir / "results" / model_key / f"e14_{args.rotation}_fold_verification.json", {
        "model": model_key, "rotation": args.rotation, "fold": fold, "audit": audit,
        "hardware": base.hardware_info(),
    })


def gptq_quantize(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E14 GPTQ requires CUDA")
    workdir = Path(args.workdir).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    output = checkpoint_dir(artifact_root, args.model, args.rotation)
    done = output / "DONE.json"
    if done.exists():
        LOG.info("E14 GPTQ checkpoint exists: %s", done)
        return
    output.mkdir(parents=True, exist_ok=True)
    base.setup_logging(workdir, f"e14-gptq-{args.model}-{args.rotation}")
    base.seed_everything(args.seed)
    model_id, model_key = act.model_id_and_key(args.model)
    tokens = base.prepare_token_chunks(
        model_id, "train", 0, args.calibration_sequences, args.seq_len, workdir
    )

    # Measure algebraic folding before GPTQ on a fixed prefix.
    probe = tokens[:1, :args.verify_tokens].cuda()
    original = base.load_model(model_id, workdir)
    with torch.inference_mode():
        reference = original(input_ids=probe, use_cache=False).logits.float()
    del original
    gc.collect()
    torch.cuda.empty_cache()
    model, rotations, fold = _prepare_rotated_model(
        workdir, model_key, args.rotation, args.seed, args.weight_row_batch
    )
    invariance = _fold_invariance(model_id, model, rotations, probe, reference)
    if invariance["relative_l2_logit_error"] > args.fold_tolerance:
        raise AssertionError(f"rotation fold failed tolerance: {invariance}")
    del reference, probe
    LOG.info("rotation fold relative logit error %.6g", invariance["relative_l2_logit_error"])

    # GPTQ sees the rotated but unquantized computation, matching QuaRot order.
    runtime = RuntimeHooks(model, rotations, activation_kind=None, quantize_kv=False)
    runtime.install()
    model.cpu()
    rotary = model.model.rotary_emb.cuda()
    layers = model.model.layers
    hidden = torch.empty(
        (tokens.shape[0], args.seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    with torch.inference_mode():
        for index in range(tokens.shape[0]):
            hidden[index] = model.model.embed_tokens(tokens[index:index + 1]).squeeze(0)
    scratch = torch.empty_like(hidden)
    position_ids = torch.arange(args.seq_len, device="cuda").unsqueeze(0)
    dummy = torch.zeros(
        (1, args.seq_len, model.config.hidden_size), device="cuda", dtype=torch.bfloat16
    )
    position_embeddings = rotary(dummy, position_ids)
    del dummy
    partial = output / "gptq_audit.partial.csv"
    audit_rows: list[dict[str, Any]] = list(base.read_csv(partial)) if partial.exists() else []
    started = time.time()
    for layer_index, layer in enumerate(layers):
        layer_path = output / f"layer_{layer_index:02d}.pt"
        if layer_path.exists():
            _load_layer_state(layer, layer_path)
            layer.cuda()
            with torch.inference_mode():
                for sequence in range(tokens.shape[0]):
                    value = _layer_forward(
                        layer, hidden[sequence:sequence + 1].cuda(), position_ids, position_embeddings
                    )
                    scratch[sequence].copy_(value.squeeze(0).cpu())
            layer.cpu()
            hidden, scratch = scratch, hidden
            LOG.info("loaded and propagated GPTQ layer %d/%d", layer_index + 1, len(layers))
            continue

        layer.cuda()
        for group_index, group in enumerate(_linear_groups(layer)):
            engines = {name: GPTQ(module) for name, module in group}
            handles = []
            for name, module in group:
                def capture(_module: torch.nn.Module, inputs: tuple[Any, ...],
                            _output: torch.Tensor, name: str = name) -> None:
                    engines[name].add_batch(inputs[0].detach())
                handles.append(module.register_forward_hook(capture))
            with torch.inference_mode():
                for sequence in range(tokens.shape[0]):
                    _layer_forward(
                        layer, hidden[sequence:sequence + 1].cuda(), position_ids, position_embeddings
                    )
            for handle in handles:
                handle.remove()
            for name, _module in group:
                audit = engines[name].fasterquant(
                    blocksize=128, percdamp=0.01, groupsize=-1
                )
                audit_rows.append({
                    "model": model_key, "rotation": args.rotation, "layer": layer_index,
                    "group": group_index, "module": name, **audit.__dict__,
                })
            del engines
            gc.collect()
            torch.cuda.empty_cache()
            LOG.info("GPTQ layer %d group %d/4 complete", layer_index, group_index + 1)

        with torch.inference_mode():
            for sequence in range(tokens.shape[0]):
                value = _layer_forward(
                    layer, hidden[sequence:sequence + 1].cuda(), position_ids, position_embeddings
                )
                scratch[sequence].copy_(value.squeeze(0).cpu())
        layer.cpu()
        base.atomic_torch_save(layer_path, _layer_state(layer))
        base.write_csv(output / "gptq_audit.partial.csv", audit_rows)
        hidden, scratch = scratch, hidden
        LOG.info("saved GPTQ layer %d/%d", layer_index + 1, len(layers))
        gc.collect()
        torch.cuda.empty_cache()

    runtime.close()
    base.write_csv(output / "gptq_audit.csv", audit_rows)
    base.atomic_json(done, {
        "model": model_key, "model_id": model_id, "rotation": args.rotation,
        "quarot_commit": QUAROT_COMMIT,
        "gptq": {
            "bits": 4, "perchannel": True, "symmetric": True, "groupsize": -1,
            "mse_clipping": True, "norm": 2.4, "grid": 100, "maxshrink": 0.8,
            "blocksize": 128, "percdamp": 0.01, "act_order": False,
            "static_groups": False, "calibration_sequences": args.calibration_sequences,
            "calibration_dataset": "WikiText-2 train", "sequence_length": args.seq_len,
            "excluded": ["embed_tokens", "lm_head"],
        },
        "fold": fold, "fold_invariance": invariance,
        "checkpoint": "one bf16 fake-quantized state file per decoder layer",
        "elapsed_seconds": time.time() - started, "hardware": base.hardware_info(),
    })
    partial.unlink(missing_ok=True)


def load_quantized_model(workdir: Path, artifact_root: Path, model_key: str,
                         rotation: str, seed: int, row_batch: int) -> tuple[torch.nn.Module, RotationSet]:
    root = checkpoint_dir(artifact_root, model_key, rotation)
    if not (root / "DONE.json").exists():
        raise FileNotFoundError(root / "DONE.json")
    model, rotations, _fold = _prepare_rotated_model(workdir, model_key, rotation, seed, row_batch)
    for index, layer in enumerate(model.model.layers):
        _load_layer_state(layer, root / f"layer_{index:02d}.pt")
    return model, rotations


def _full_wikitext_tokens(model_id: str, workdir: Path, seq_len: int) -> torch.Tensor:
    path = workdir / "cache" / "tokenized" / f"{base.model_key_from_id(model_id)}-wikitext2-test-full-l{seq_len}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
        cache_dir=str(workdir / "cache" / "datasets"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=str(workdir / "cache" / "huggingface"), use_fast=True
    )
    text = "\n\n".join(row["text"] for row in dataset)
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    chunks = torch.tensor(ids[:len(ids) // seq_len * seq_len], dtype=torch.long).reshape(-1, seq_len)
    base.atomic_torch_save(path, chunks)
    return chunks


def _row_settings(row: str) -> tuple[str, str]:
    if row == "quarot":
        return "hadamard", "quarot_symmetric_token"
    if row == "hadamard_asym_g128":
        return "hadamard", "asymmetric_g128"
    if row == "nar_asym_g128":
        return "nar", "asymmetric_g128"
    raise ValueError(row)


def _evaluate_ppl(model: torch.nn.Module, tokens: torch.Tensor, label: str) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    with torch.inference_mode():
        for index in range(tokens.shape[0]):
            batch = tokens[index:index + 1].cuda()
            output = model(input_ids=batch, labels=batch, use_cache=False)
            nll = float(output.loss.float())
            if not math.isfinite(nll):
                raise RuntimeError(f"non-finite E14 loss at chunk {index}")
            rows.append({"chunk": index, "nll": nll, "tokens_scored": tokens.shape[1] - 1})
            if index % 16 == 0:
                LOG.info("%s PPL chunk %d/%d nll=%.6f", label, index + 1, tokens.shape[0], nll)
    ppl = math.exp(float(np.average(
        [row["nll"] for row in rows], weights=[row["tokens_scored"] for row in rows]
    )))
    return ppl, rows


def evaluate_row(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E14 evaluation requires CUDA")
    workdir = Path(args.workdir).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    result_dir = workdir / "results" / args.model
    ppl_path = result_dir / f"e14_{args.row}_ppl.json"
    zero_path = result_dir / f"e14_{args.row}_zero_shot.json"
    if ppl_path.exists() and zero_path.exists():
        LOG.info("E14 row exists: %s %s", ppl_path, zero_path)
        return
    base.setup_logging(workdir, f"e14-evaluate-{args.model}-{args.row}")
    base.seed_everything(args.seed)
    rotation, activation_kind = _row_settings(args.row)
    model_id, model_key = act.model_id_and_key(args.model)
    model, rotations = load_quantized_model(
        workdir, artifact_root, model_key, rotation, args.seed, args.weight_row_batch
    )
    runtime = RuntimeHooks(model, rotations, activation_kind=activation_kind, quantize_kv=True)
    runtime.install()
    try:
        if not ppl_path.exists():
            tokens = _full_wikitext_tokens(model_id, workdir, args.seq_len)
            ppl, chunk_rows = _evaluate_ppl(model, tokens, f"{model_key} {args.row}")
            base.atomic_json(ppl_path, {
                "model": model_key, "model_id": model_id, "row": args.row,
                "rotation_checkpoint": rotation, "ppl": ppl, "chunks": chunk_rows,
                "dataset": "WikiText-2 raw test full contiguous token stream",
                "sequence_length": args.seq_len, "chunks_evaluated": len(chunk_rows),
                "seed": args.seed, "hardware": base.hardware_info(),
            })
            del tokens
        if not zero_path.exists():
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            from lm_eval.tasks import TaskManager
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, cache_dir=str(workdir / "cache" / "huggingface"), use_fast=True
            )
            lm = HFLM(
                pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size,
                max_batch_size=args.batch_size, max_length=args.seq_len,
            )
            result = lm_eval.simple_evaluate(
                model=lm, tasks=list(TASKS), num_fewshot=0,
                batch_size=args.batch_size, max_batch_size=args.batch_size,
                task_manager=TaskManager(), cache_requests=False, bootstrap_iters=0,
                log_samples=False, random_seed=args.seed, numpy_random_seed=args.seed,
                torch_random_seed=args.seed, fewshot_random_seed=args.seed,
                apply_chat_template=False, fewshot_as_multiturn=False,
            )
            if result is None:
                raise RuntimeError("lm-eval returned no result")
            task_rows = [
                {"task": task, "metric": METRICS[task],
                 "accuracy": float(result["results"][task][METRICS[task]])}
                for task in TASKS
            ]
            base.atomic_json(zero_path, {
                "model": model_key, "model_id": model_id, "row": args.row,
                "rotation_checkpoint": rotation, "tasks": task_rows,
                "mean_accuracy": float(np.mean([row["accuracy"] for row in task_rows])),
                "mean_definition": "unweighted mean of six frozen accuracy metrics",
                "seed": args.seed, "num_fewshot": 0, "harness_commit": HARNESS_COMMIT,
                "task_versions": result.get("versions", {}),
                "sample_counts": result.get("n-samples", {}),
                "hardware": base.hardware_info(),
            })
            del lm
    finally:
        runtime.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def finalize(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    rows = []
    for model in MODELS:
        for row_name in ROWS:
            root = workdir / "results" / model
            ppl = _json(root / f"e14_{row_name}_ppl.json")
            zero = _json(root / f"e14_{row_name}_zero_shot.json")
            values = {item["task"]: item["accuracy"] for item in zero["tasks"]}
            rows.append({
                "model": model, "row": row_name, "ppl": ppl["ppl"],
                **{task: values[task] for task in TASKS},
                "mean_accuracy": zero["mean_accuracy"], "seed": zero["seed"],
            })
    base.write_csv(workdir / "results" / "e14_w4a4kv4_summary.csv", rows)
    base.atomic_json(workdir / "results" / "E14_DONE.json", {
        "models": list(MODELS), "rows": list(ROWS), "seed": args.seed,
        "paired": "same GPTQ calibration sequences, WikiText-2 full-test tokens, harness revision, tasks, and seed",
        "w4": {
            "source": f"spcl/QuaRot@{QUAROT_COMMIT}", "method": "GPTQ",
            "groupsize": -1, "symmetric": True, "mse_clipping": True,
            "percdamp": 0.01, "act_order": False,
        },
        "a4": {
            "quarot": "official symmetric per-token semantics",
            "other_rows": "dynamic asymmetric per-token group-128 with fp16 scale and fp16 offset",
            "sites": "inputs to q/k/v/o/gate/up/down projections",
        },
        "k4": {
            "all_rows": "post-RoPE KIVI-style dynamic asymmetric per-channel, contiguous token groups of 32",
            "residual": "incomplete newest group of fewer than 32 tokens remains bf16",
            "r3": "omitted because the fixed per-channel K baseline replaces QuaRot per-token rotated K",
        },
        "v4": {
            "all_rows": "dynamic asymmetric per-token, one head_dim=128 group",
            "quarot_and_hadamard": "Hadamard R2", "nar": "NAR R2 folded into o_proj",
        },
        "single_seed": True, "no_tuning": True, "negative_results_reported": True,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--artifact-root", default=str(Path.home() / "nar-e14-artifacts"))
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--weight-row-batch", type=int, default=256)
    parser.add_argument("--fold-tolerance", type=float, default=0.02,
                        help="pre-registered relative-L2 logit gate before GPTQ")
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--model", choices=MODELS, required=True)
    cal.add_argument("--calibration-sequences", type=int, default=128)
    verify = sub.add_parser("verify")
    verify.add_argument("--model", choices=MODELS, required=True)
    verify.add_argument("--rotation", choices=ROTATIONS, required=True)
    verify.add_argument("--verify-tokens", type=int, default=128)
    gptq = sub.add_parser("gptq")
    gptq.add_argument("--model", choices=MODELS, required=True)
    gptq.add_argument("--rotation", choices=ROTATIONS, required=True)
    gptq.add_argument("--calibration-sequences", type=int, default=128)
    gptq.add_argument("--verify-tokens", type=int, default=128)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", choices=MODELS, required=True)
    evaluate.add_argument("--row", choices=ROWS, required=True)
    evaluate.add_argument("--batch-size", type=int, default=2)
    sub.add_parser("finalize")
    return parser


def main() -> None:
    args = parser().parse_args()
    {
        "calibrate": calibrate_rotations, "verify": verify_rotation,
        "gptq": gptq_quantize, "evaluate": evaluate_row, "finalize": finalize,
    }[args.command](args)


if __name__ == "__main__":
    main()
