"""E26 — Qwen3-30B-A3B-Base: the method on a mixture of experts.

The dense Qwen3 pipeline (E22) with four MoE-specific adaptations:

1. Router-preserving fold. R1 is folded into the router's input columns like
   every other consumer of the residual, so routing logits are unchanged in
   exact arithmetic. The router reads the UNQUANTIZED post-norm activation
   (the shared expert input is quantized after routing), so no routing
   decision depends on the activation quantizer. ``control`` audits the
   routing-agreement rate per layer (top-8 set and argmax) between the
   original and the rotated unquantized model.
2. Expert-level null space. Each expert's down_proj input is 768-dim: six
   group-128 slots, so R4 is per expert with k <= 6 (both k=8 and k=max rows
   use 6 there; they differ in R1). ``calibrate`` reports f at k=6 per expert
   and the sqrt(1-f) range law on held-out routed rows.
3. Calibration for cold experts. Sigma_e from an expert's own routed
   calibration tokens when n_e >= N0 = 2048; below that the layer-pooled
   Sigma is a shrinkage prior, Sigma_e <- (n_e Sigma_e + N0 Sigma_pool)/(n_e + N0).
   Variants for the ablation: ``per_expert`` (Hadamard-only R4 for cold
   experts), ``pooled`` (one R4 per layer), ``shrinkage`` (main).
4. Per-expert GPTQ. gate_up and down of every expert are quantized with the
   Hessian of its own routed tokens; below GPTQ_MIN = 512 routed tokens the
   layer-pooled Hessian is used (rotated into the expert's R4 basis for the
   down site). Per-expert relative weight error and routed-token count are
   recorded together.

Everything else is E22's: attention R1/R2, K per-channel and V per-token
(KIVI), GPTQ g128_asym, asymmetric group-128 activations, fp32 containers,
fp32 NLL. The model is sharded in fp32 across the visible GPUs for
calibration, control and evaluation and held on the CPU for the GPTQ layer
loop, as the 70B was (E21).

Artifacts: results/qwen3_30b_a3b_base/e26_*.json|csv,
activations/qwen3_30b_a3b_base/e26_factors/<variant>/layer_XX_expert_YYY.pt,
artifacts/e26/qwen3_30b_a3b_base/gptq_<rotation>_seed0_g128_asym[_<variant>]/layer_XX.pt.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nar import activation_experiments as act  # noqa: E402
from nar import e12_wy  # noqa: E402
from nar import e14_w4a4kv4 as e14  # noqa: E402
from nar import e19_qwen3_e2e as e19  # noqa: E402
from nar import e18_v2 as e18v2
from nar import e21_llama70b_e2e as e21  # noqa: E402
from nar import e22_qwen3_family as e22  # noqa: E402
from nar import experiment as base  # noqa: E402
from nar import quarot_gptq  # noqa: E402

LOG = logging.getLogger("nar.e26")
WORKDIR = Path(".")
MODEL_KEY = "qwen3_30b_a3b_base"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Base"
PREFIX = "e26"
GROUP = e14.GROUP
PROTOCOL = "g128_asym"
ACTIVATION_KIND = "asymmetric_g128"
N0 = 2048          # shrinkage prior weight (routed tokens)
GPTQ_MIN = 512     # below this, the pooled Hessian
VARIANTS = ("shrinkage", "per_expert", "pooled")
MAIN_VARIANT = "shrinkage"
ROTATIONS = ("hadamard", "nar_k8", "nar_kmax")
ROWS = {"bf16": None, "hadamard_asym_g128": "hadamard",
        "nar_k8_asym_g128": "nar_k8", "nar_kmax_asym_g128": "nar_kmax"}
R1_LABEL = {"nar_k8": "k8", "nar_kmax": "kmax"}
EXPECTED = {
    "num_hidden_layers": 48, "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
    "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768,
}
HELD_OUT_ROWS = 256   # routed rows kept per expert for the sqrt(1-f) range law


# ------------------------------------------------------------------ paths ---

def result_dir() -> Path:
    path = WORKDIR / "results" / MODEL_KEY
    path.mkdir(parents=True, exist_ok=True)
    return path


def factor_root(variant: str) -> Path:
    return WORKDIR / "activations" / MODEL_KEY / "e26_factors" / variant


def artifact_root() -> Path:
    return WORKDIR / "artifacts" / "e26"


def checkpoint_dir(rotation: str, seed: int, variant: str) -> Path:
    suffix = "" if variant == MAIN_VARIANT else f"_{variant}"
    return artifact_root() / MODEL_KEY / f"gptq_{rotation}_seed{seed}_{PROTOCOL}{suffix}"


# ---------------------------------------------------------------- loading ---

def load_sharded(dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM
    memory = e21._max_memory()
    LOG.info("E26 loading %s in %s across %d GPUs: %s", MODEL_ID, dtype, len(memory), memory)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=str(WORKDIR / "cache" / "huggingface"), dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="sdpa", device_map="balanced", max_memory=memory)
    install_experts_patch()
    return model.eval()


def load_cpu() -> torch.nn.Module:
    model = e21.load_model_cpu(MODEL_ID, WORKDIR)
    install_experts_patch()
    return model


def moe_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [block for block in model.model.layers]


# ------------------------------------------------------- experts forward ---

class ExpertContext:
    """What the patched Qwen3MoeExperts.forward does around each expert.

    input_fn      applied once to the shared post-norm input (already routed,
                  so the router never sees it): the activation quantizer.
    down_fn(e, h) applied to expert e's 768-dim down input: R4_e, then the
                  activation quantizer.
    gate_up_capture(e, x_e) / down_capture(e, h_e): GPTQ Hessians and the
                  calibration collectors.
    """

    def __init__(self, layer: int, input_fn: Callable | None = None, down_fn: Callable | None = None,
                 gate_up_capture: Callable | None = None, down_capture: Callable | None = None,
                 pooled_capture: Callable | None = None):
        self.layer = layer
        self.input_fn = input_fn
        self.down_fn = down_fn
        self.gate_up_capture = gate_up_capture
        self.down_capture = down_capture
        self.pooled_capture = pooled_capture

    def forward(self, module: torch.nn.Module, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        x = hidden_states if self.input_fn is None else self.input_fn(hidden_states)
        if self.pooled_capture is not None:
            self.pooled_capture(x)
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=module.num_experts).permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            e = int(expert_idx[0])
            if e == module.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current = x[token_idx]
            if self.gate_up_capture is not None:
                self.gate_up_capture(e, current)
            gate, up = torch.nn.functional.linear(current, module.gate_up_proj[e]).chunk(2, dim=-1)
            h = module.act_fn(gate) * up
            if self.down_fn is not None:
                h = self.down_fn(e, h)
            if self.down_capture is not None:
                self.down_capture(e, h)
            out = torch.nn.functional.linear(h, module.down_proj[e])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final


_PATCHED = False


def install_experts_patch() -> None:
    """Route Qwen3MoeExperts.forward through the module's ExpertContext when one is set."""
    global _PATCHED
    if _PATCHED:
        return
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
    original = Qwen3MoeExperts.forward

    def forward(self, hidden_states, top_k_index, top_k_weights):  # type: ignore[no-untyped-def]
        ctx = getattr(self, "_nar_ctx", None)
        if ctx is None:
            return original(self, hidden_states, top_k_index, top_k_weights)
        return ctx.forward(self, hidden_states, top_k_index, top_k_weights)

    Qwen3MoeExperts.forward = forward  # type: ignore[assignment]
    Qwen3MoeExperts._nar_original_forward = original  # type: ignore[attr-defined]
    _PATCHED = True


# ------------------------------------------------------------------ audit ---

def architecture_audit(model: torch.nn.Module) -> dict[str, Any]:
    config = model.config
    problems: list[str] = []
    fields = {
        "num_hidden_layers": int(config.num_hidden_layers), "hidden_size": int(config.hidden_size),
        "num_attention_heads": int(config.num_attention_heads), "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "num_experts": int(config.num_experts), "num_experts_per_tok": int(config.num_experts_per_tok),
        "moe_intermediate_size": int(config.moe_intermediate_size),
    }
    for key, want in EXPECTED.items():
        if fields[key] != want:
            problems.append(f"{key}={fields[key]} expected {want}")
    if config.architectures[0] != "Qwen3MoeForCausalLM":
        problems.append(f"architecture {config.architectures[0]}")
    if int(config.decoder_sparse_step) != 1 or list(config.mlp_only_layers or []):
        problems.append(f"not every layer is MoE: sparse_step={config.decoder_sparse_step} mlp_only={config.mlp_only_layers}")
    if getattr(config, "shared_expert_intermediate_size", 0) or any(
            hasattr(b.mlp, "shared_expert") for b in model.model.layers):
        problems.append("a shared expert is present")
    if getattr(config, "tie_word_embeddings", False):
        problems.append("tied embeddings")
    if getattr(config, "attention_bias", False):
        problems.append("attention bias")
    biased = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m.bias is not None]
    if biased:
        problems.append(f"biased linear modules: {biased[:4]}")
    qk = sum(1 for n, _ in model.named_modules() if n.endswith(("q_norm", "k_norm")))
    if qk != 2 * fields["num_hidden_layers"]:
        problems.append(f"q_norm/k_norm count {qk}")
    block = model.model.layers[0]
    router = block.mlp.gate
    if tuple(router.weight.shape) != (fields["num_experts"], fields["hidden_size"]) or getattr(router, "bias", None) is not None:
        problems.append(f"router is not one bias-free linear gate: {tuple(router.weight.shape)}")
    experts = block.mlp.experts
    if tuple(experts.gate_up_proj.shape) != (128, 2 * 768, 2048) or tuple(experts.down_proj.shape) != (128, 2048, 768):
        problems.append(f"expert parameter shapes {tuple(experts.gate_up_proj.shape)} {tuple(experts.down_proj.shape)}")
    audit = {
        "model_id": MODEL_ID, "architecture": config.architectures[0], "fields": fields,
        "norm_topk_prob": bool(config.norm_topk_prob), "decoder_sparse_step": int(config.decoder_sparse_step),
        "mlp_only_layers": list(config.mlp_only_layers or []), "shared_expert": False,
        "router": "one linear gate (128 x 2048, no bias) on the post-norm hidden state; softmax, top-8, renormalised" if config.norm_topk_prob else "one linear gate on the post-norm hidden state; softmax, top-8",
        "expert_parameters": "fused 3-D tensors gate_up_proj [128, 1536, 2048] and down_proj [128, 2048, 768] (transformers 5.x)",
        "slot_counts": {"residual (R1)": fields["hidden_size"] // GROUP, "expert down input (R4)": fields["moe_intermediate_size"] // GROUP},
        "hadamard_orders": {"2048": "2^11", "768": "12 x 64 (Paley 12)", "heads 32": "2^5"},
        "tie_word_embeddings": False, "linear_modules_with_bias": biased, "vocab_size": int(config.vocab_size),
        "problems": problems,
    }
    if problems:
        raise AssertionError("E26 architecture audit failed:\n  " + "\n  ".join(problems))
    return audit


def audit_command(args: argparse.Namespace) -> None:
    base.setup_logging(WORKDIR, "e26-audit")
    model = load_sharded()
    audit = architecture_audit(model)
    audit["compute_dtype"] = "float32"
    audit["git_commit"] = e19.git_commit()
    audit["hardware"] = base.hardware_info()
    base.atomic_json(result_dir() / f"{PREFIX}_architecture_audit.json", audit)
    LOG.info("E26 audit: problems=%s", audit["problems"])


# ------------------------------------------------------------ rotations ---

class MoERotationSet(e14.RotationSet):
    """R1 (global) and R2 (per head) from E14's calibration; R4 per (layer, expert)."""

    def __init__(self, workdir: Path, model_key: str, method: str, seed: int, config: Any,
                 device: torch.device, variant: str = MAIN_VARIANT):
        self.method = method
        self.seed = seed
        self.device = device
        self.variant = variant
        self.layers = int(config.num_hidden_layers)
        self.heads = int(config.num_attention_heads)
        self.hidden = int(config.hidden_size)
        self.intermediate = int(config.moe_intermediate_size)
        self.experts = int(config.num_experts)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.r1 = None
        self.r2: dict[int, act.RotationFactor] = {}
        self.r4: dict[int, act.RotationFactor] = {}
        self.r4_wy: dict[int, e12_wy.WYFactor] = {}
        self.expert_r4: dict[tuple[int, int], act.RotationFactor] = {}
        self.expert_wy: dict[tuple[int, int], e12_wy.WYFactor] = {}
        self._dev_r1: dict[torch.device, act.RotationFactor] = {}
        self._dev_r2: dict[tuple[torch.device, int], act.RotationFactor] = {}
        self._dev_expert: dict[tuple[torch.device, int, int], e12_wy.WYFactor] = {}
        self._signs: dict[tuple[str, int, torch.device], torch.Tensor] = {}
        self.sign_cache: dict[tuple[str, int], torch.Tensor] = {}
        if not method.startswith("nar_"):
            return
        root = e14.rotation_dir(workdir, model_key, seed)
        if not (root / "DONE.json").exists():
            raise FileNotFoundError(root / "DONE.json")
        self.r1 = act.RotationFactor.load(root / f"r1_{R1_LABEL[method]}.pt", device)
        froot = factor_root(variant)
        if not (froot / "DONE.json").exists():
            raise FileNotFoundError(froot / "DONE.json")
        for layer in range(self.layers):
            self.r2[layer] = act.RotationFactor.load(root / f"r2_v_layer_{layer:02d}.pt", device)
            for e in range(self.experts):
                factor = act.RotationFactor.load(froot / f"layer_{layer:02d}_expert_{e:03d}.pt", torch.device("cpu"))
                self.expert_r4[(layer, e)] = factor

    def ranks(self) -> dict[str, Any]:
        r4 = [int(f.active.sum()) for f in self.expert_r4.values()] if self.expert_r4 else []
        return {"r1_rank": int(self.r1.active.sum()) if self.r1 is not None else None,
                "r1_slots": self.hidden // GROUP,
                "r4_rank": (max(r4) if r4 else None), "r4_rank_min": (min(r4) if r4 else None),
                "r4_slots": self.intermediate // GROUP, "r4_per_expert": True,
                "r2_rank": int(self.r2[0].active.sum()) if self.r2 else None}

    def _signs_on(self, label: str, layer: int, n: int, device: torch.device) -> torch.Tensor:
        key = (label, layer, device)
        if key not in self._signs:
            self._signs[key] = e14._signs(n, e14._seed(self.seed, label, layer), device)
        return self._signs[key]

    def signs(self, label: str, layer: int, n: int) -> torch.Tensor:
        return self._signs_on(label, layer, n, self.device)

    def apply(self, label: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        device = value.device
        signs = self._signs_on(label, layer, value.shape[-1], device)
        if self.method == "hadamard":
            if label != "r1":
                signs = torch.ones_like(signs)
            return act.full_hadamard_rows(value.float(), signs)
        if label == "r1":
            if device not in self._dev_r1:
                self._dev_r1[device] = e21._factor_to(self.r1, device)
            return self._dev_r1[device].apply(value, signs)
        if label == "r2":
            key = (device, layer)
            if key not in self._dev_r2:
                self._dev_r2[key] = e21._factor_to(self.r2[layer], device)
            return self._dev_r2[key].apply(value, signs)
        raise KeyError(f"E26 has no dense R4; use apply_expert for label {label}")

    def apply_expert_transpose(self, layer: int, expert: int, value: torch.Tensor) -> torch.Tensor:
        """R4_e^T, so apply_expert_transpose(apply_expert(x)) == x in exact arithmetic."""
        device = value.device
        signs = torch.ones(value.shape[-1], device=device, dtype=torch.float32)
        if self.method == "hadamard":
            return e18v2.full_hadamard_rows_transpose(value.float(), signs)
        factor = self.expert_r4[(layer, expert)]
        n = factor.n
        shape = value.shape
        rows = value.float().reshape(-1, n)
        rows = act.ext._fast_walsh_hadamard(rows.reshape(-1, n // factor.b, factor.b)).reshape(-1, n)
        rows = rows * signs.to(rows.device)
        source = factor.source_order.to(rows.device)
        target = factor.target_order.to(rows.device)
        unpermuted = torch.empty_like(rows)
        unpermuted[:, source] = rows[:, target]
        if not bool(factor.active.any()):
            return unpermuted.reshape(shape)
        w, y = e12_wy.compact_wy(factor.reflectors.to(rows.device), factor.active.to(rows.device))
        return (unpermuted - (unpermuted @ y) @ w.T).reshape(shape)

    def apply_expert(self, layer: int, expert: int, value: torch.Tensor) -> torch.Tensor:
        """R4 of one expert on its 768-dim down input (rows of value)."""
        device = value.device
        signs = torch.ones(value.shape[-1], device=device, dtype=torch.float32)  # NAR/Hadamard R4 rows carry no signs
        if self.method == "hadamard":
            return act.full_hadamard_rows(value.float(), signs)
        key = (device, layer, expert)
        if key not in self._dev_expert:
            factor = e21._factor_to(self.expert_r4[(layer, expert)], device)
            if bool(factor.active.any()):
                w, y = e12_wy.compact_wy(factor.reflectors, factor.active)
                self._dev_expert[key] = e12_wy.WYFactor(factor, w, y)
            else:
                self._dev_expert[key] = factor  # Hadamard-only cold expert: no reflectors
        return self._dev_expert[key].apply(value, signs)


# ------------------------------------------------------------------- fold ---

def _transform_rows(param: torch.Tensor, transform: Callable[[torch.Tensor], torch.Tensor], row_batch: int) -> None:
    """In place: every row of a 2-D tensor through ``transform`` (input-axis fold)."""
    home = param.device
    for start in range(0, param.shape[0], row_batch):
        chunk = param[start:start + row_batch].float()
        if e14.FOLD_DEVICE is not None:
            chunk = chunk.to(e14.FOLD_DEVICE)
        param[start:start + row_batch].copy_(transform(chunk).to(param.dtype).to(home))


def _transform_left(param: torch.Tensor, transform: Callable[[torch.Tensor], torch.Tensor], row_batch: int) -> None:
    """In place: output-axis fold Q^T W of a 2-D tensor."""
    transposed = param.T.contiguous()
    _transform_rows(transposed, transform, row_batch)
    param.copy_(transposed.T)


def fuse_norms_and_rotate_moe(model: torch.nn.Module, rotations: MoERotationSet, row_batch: int) -> dict[str, Any]:
    """E14's fold with the expert tensors and the router as residual consumers."""
    with torch.no_grad():
        for block in model.model.layers:
            input_scale = block.input_layernorm.weight.detach().float()
            post_scale = block.post_attention_layernorm.weight.detach().float()
            for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj):
                module.weight.mul_(input_scale.to(module.weight.dtype).unsqueeze(0))
            experts = block.mlp.experts
            experts.gate_up_proj.data.mul_(post_scale.to(experts.gate_up_proj.dtype).to(experts.gate_up_proj.device).view(1, 1, -1))
            router = block.mlp.gate
            router.weight.data.mul_(post_scale.to(router.weight.dtype).to(router.weight.device).unsqueeze(0))
            block.input_layernorm.weight.fill_(1)
            block.post_attention_layernorm.weight.fill_(1)
        final_scale = model.model.norm.weight.detach().float()
        model.lm_head.weight.mul_(final_scale.to(model.lm_head.weight.dtype).unsqueeze(0))
        model.model.norm.weight.fill_(1)

        r1 = lambda value: rotations.apply("r1", 0, value)  # noqa: E731
        e14._transform_weight_rows(model.model.embed_tokens, r1, row_batch)
        e14._transform_weight_rows(model.lm_head, r1, row_batch)
        heads, head_dim = rotations.heads, rotations.head_dim
        for layer, block in enumerate(model.model.layers):
            for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj):
                e14._transform_weight_rows(module, r1, row_batch)
            e14._transform_weight_left(block.self_attn.o_proj, r1, row_batch)
            experts = block.mlp.experts
            _transform_rows(block.mlp.gate.weight.data, r1, row_batch)
            for e in range(rotations.experts):
                _transform_rows(experts.gate_up_proj.data[e], r1, row_batch)          # [1536, 2048]: input axis
                _transform_left(experts.down_proj.data[e], r1, row_batch)             # [2048, 768]: output axis
                _transform_rows(experts.down_proj.data[e],
                                lambda v, layer=layer, e=e: rotations.apply_expert(layer, e, v), row_batch)  # input axis 768
            weight = block.self_attn.o_proj.weight.detach()
            shaped = weight.reshape(weight.shape[0], heads, head_dim)
            rotated = rotations.apply("r2", layer, shaped.reshape(-1, head_dim)).reshape_as(shaped)
            rotated = rotations.apply_r3(rotated.reshape_as(weight)).to(weight.dtype)
            weight.copy_(rotated.reshape_as(weight))
            LOG.info("E26 fold layer %d/%d", layer + 1, rotations.layers)
    return {
        "r1": "global residual rotation; folded into q/k/v, every expert's gate_up input axis, the router's input axis, "
              "and the output axes of o_proj and every expert's down_proj",
        "router": "R1 folded into the gate's input columns; routing logits unchanged in exact arithmetic; router kept unquantized",
        "r2": "per-head V rotation with identical fold into every GQA-expanded o_proj head block",
        "r4": "per-expert down-input rotation folded into that expert's down_proj rows",
        "norm": "input/post/final RMSNorm affine weights fused into consumers (q/k/v; gate_up and router; lm_head), then set to one",
    }


# ---------------------------------------------------------------- hooks ---

class MoERuntimeHooks(e14.RuntimeHooks):
    """E14's hooks with the expert contexts in place of the dense MLP hooks."""

    def __init__(self, model: torch.nn.Module, rotations: MoERotationSet, activation_kind: str | None,
                 quantize_kv: bool = True, round_trip: bool = False):
        super().__init__(model, rotations, activation_kind, quantize_kv)
        self.rotations: MoERotationSet = rotations
        # round_trip installs R4_e followed by its transpose at the expert
        # input and leaves every weight untouched: mathematically the
        # identity, numerically the same reordering the fold introduces.  It
        # measures the fp32 noise floor of the routing-agreement audit.
        self.round_trip = round_trip

    def _quantize(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation_kind == "asymmetric_g128":
            return base.dynamic_asym_int4(value, GROUP)[0].to(value.dtype)
        if self.activation_kind == "symmetric_g128":
            return e14._symmetric_per_group_int4(value, GROUP).to(value.dtype)
        return value

    def install(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if self.quantize_kv:
            ALL_ATTENTION_FUNCTIONS.register(self.attention_key, self.attention)
            try:
                from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
                ALL_MASK_ATTENTION_FUNCTIONS.register(self.attention_key, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
            except (ImportError, KeyError):
                pass
            self.model.config._attn_implementation = self.attention_key
        quantize = self.activation_kind is not None
        for layer, block in enumerate(self.model.model.layers):
            # The attention rotations are only correct against folded weights:
            # rotate_v feeds o_proj a rotated V, which the fold has undone in
            # o_proj's rows.  The null probe leaves every weight alone, so it
            # installs the expert round trip and nothing else.
            if not self.round_trip:
                self.handles.append(block.self_attn.v_proj.register_forward_hook(self.rotate_v(layer)))
                if self.rotations.method == "hadamard":
                    self.handles.append(block.self_attn.o_proj.register_forward_pre_hook(self.rotate_o()))
            if quantize:
                for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj,
                               block.self_attn.o_proj):
                    self.handles.append(module.register_forward_pre_hook(self.quantize_input))

            def down_fn(e: int, h: torch.Tensor, layer: int = layer) -> torch.Tensor:
                rotated = self.rotations.apply_expert(layer, e, h).to(h.dtype)
                if self.round_trip:
                    # Null probe: undo the rotation immediately, so the weights
                    # stay unfolded and the arithmetic is an exact identity.
                    return self.rotations.apply_expert_transpose(layer, e, rotated).to(h.dtype)
                return self._quantize(rotated) if quantize else rotated

            block.mlp.experts._nar_ctx = ExpertContext(
                layer, input_fn=(self._quantize if quantize else None), down_fn=down_fn)

    def close(self) -> None:
        super().close()
        for block in self.model.model.layers:
            if hasattr(block.mlp.experts, "_nar_ctx"):
                del block.mlp.experts._nar_ctx


# ------------------------------------------------------------ calibration ---

def calibration_tokens(args: argparse.Namespace) -> torch.Tensor:
    return base.prepare_token_chunks(MODEL_ID, "train", 0, args.calibration_sequences, args.seq_len, WORKDIR)


class ExpertCovarianceCollector:
    """Per-expert Sigma_e and n_e of the down input, the layer-pooled Sigma, and
    a held-out sample of routed rows per expert."""

    def __init__(self, model: torch.nn.Module, held_out: int = HELD_OUT_ROWS):
        self.model = model
        self.layers = len(model.model.layers)
        self.experts = int(model.config.num_experts)
        self.dim = int(model.config.moe_intermediate_size)
        self.sigma: dict[int, torch.Tensor] = {}
        self.count: dict[int, torch.Tensor] = {}
        self.rows: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.held_out = held_out
        self.seen_tokens = 0

    def _capture(self, layer: int) -> Callable:
        def capture(e: int, h: torch.Tensor) -> None:
            rows = h.detach().float()
            if layer not in self.sigma:
                self.sigma[layer] = torch.zeros((self.experts, self.dim, self.dim), dtype=torch.float64, device=rows.device)
                self.count[layer] = torch.zeros(self.experts, dtype=torch.long, device=rows.device)
            self.sigma[layer][e] += (rows.double().T @ rows.double())
            self.count[layer][e] += rows.shape[0]
            kept = self.rows.setdefault((layer, e), [])
            have = sum(r.shape[0] for r in kept)
            if have < self.held_out:
                kept.append(rows[: self.held_out - have].cpu())
        return capture

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers):
            block.mlp.experts._nar_ctx = ExpertContext(layer, down_capture=self._capture(layer))

    def close(self) -> None:
        for block in self.model.model.layers:
            if hasattr(block.mlp.experts, "_nar_ctx"):
                del block.mlp.experts._nar_ctx


def _hadamard_only_factor(n: int) -> act.RotationFactor:
    """R4 with no reflectors: a per-group Hadamard with the identity order."""
    return act.RotationFactor(n=n, b=GROUP, reflectors=torch.zeros((n // GROUP, n)), active=torch.zeros(n // GROUP, dtype=torch.bool),
                              source_order=torch.arange(n), target_order=torch.arange(n), anchor_error=0.0)


def _factor_from_sigma(sigma: torch.Tensor, rank: int) -> tuple[act.RotationFactor, dict[str, Any]]:
    """Top-``rank`` eigenvectors of Sigma as sequential reflectors; orders from
    the post-reflector coordinate energies diag(G Sigma G^T)."""
    n = sigma.shape[0]
    sym = (sigma + sigma.T) / 2
    values, vectors = torch.linalg.eigh(sym)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    top = vectors[:, order[:rank]].float()
    reflectors, active, error = act.reflectors_from_vectors(top, GROUP)
    # apply_reflectors maps row vectors x -> x M; the coordinate energies after
    # it are diag(M^T Sigma M), with M = apply_reflectors(I).
    m = act.apply_reflectors(torch.eye(n, dtype=torch.float32), reflectors, active)
    energies = torch.einsum("ji,jk,ki->i", m, sym.float(), m)
    source, target = act.balanced_orders(energies.clamp_min(0).sqrt().unsqueeze(0), rank, GROUP)
    factor = act.RotationFactor(n=n, b=GROUP, reflectors=reflectors, active=active,
                                source_order=source, target_order=target, anchor_error=error)
    trace = float(values.sum())
    f = float(values[:rank].sum() / trace) if trace > 0 else 0.0
    return factor, {"eigenvalues": values[:rank].tolist(), "trace": trace, "f_at_rank": f}


def calibrate_command(args: argparse.Namespace) -> None:
    """R1/R2 through E14's calibration (bf16 model, sharded); expert R4 for all variants."""
    base.setup_logging(WORKDIR, "e26-calibrate")
    base.seed_everything(args.seed)
    # R1 (k8, kmax) and R2 via E14 on a sharded bf16 model.
    args.model = MODEL_KEY
    base.load_model = lambda model_id, workdir: load_sharded(torch.bfloat16)  # E14's R1/R2 pass on the sharded bf16 model
    e14.calibrate_rotations(args)
    gc.collect(); torch.cuda.empty_cache()
    done_all = all((factor_root(v) / "DONE.json").exists() for v in VARIANTS)
    if done_all:
        LOG.info("E26 expert factors exist")
        return
    tokens = calibration_tokens(args)
    model = load_sharded()
    collector = ExpertCovarianceCollector(model)
    collector.install()
    try:
        with torch.inference_mode():
            for index in range(tokens.shape[0]):
                model.model(input_ids=tokens[index:index + 1].to(model.get_input_embeddings().weight.device), use_cache=False)
                if index % 8 == 0:
                    LOG.info("E26 expert covariance pass %d/%d", index + 1, tokens.shape[0])
    finally:
        collector.close()
    layers, experts, dim = collector.layers, collector.experts, collector.dim
    rank = dim // GROUP
    count_rows: list[dict[str, Any]] = []
    f_rows: list[dict[str, Any]] = []
    law_rows: list[dict[str, Any]] = []
    fallback = {v: {"own": 0, "prior": 0, "hadamard": 0, "pooled": 0} for v in VARIANTS}
    for layer in range(layers):
        sigma = collector.sigma[layer].cpu()
        counts = collector.count[layer].cpu()
        pooled = sigma.sum(0)
        n_pool = int(counts.sum())
        pooled_mean = pooled / max(n_pool, 1)
        pooled_factor, pooled_meta = _factor_from_sigma(pooled_mean, rank)
        for e in range(experts):
            n_e = int(counts[e])
            own = sigma[e] / max(n_e, 1)
            count_rows.append({"layer": layer, "expert": e, "routed_tokens": n_e,
                               "share_of_tokens": n_e / max(collector_total(counts), 1)})
            for variant in VARIANTS:
                if variant == "pooled":
                    factor, meta = pooled_factor, dict(pooled_meta); fallback[variant]["pooled"] += 1; source = "pooled"
                elif variant == "per_expert":
                    if n_e >= N0:
                        factor, meta = _factor_from_sigma(own, rank); fallback[variant]["own"] += 1; source = "own"
                    else:
                        factor, meta = _hadamard_only_factor(dim), {"eigenvalues": [], "trace": float(own.trace()), "f_at_rank": 0.0}
                        fallback[variant]["hadamard"] += 1; source = "hadamard_only"
                else:  # shrinkage
                    if n_e >= N0:
                        factor, meta = _factor_from_sigma(own, rank); fallback[variant]["own"] += 1; source = "own"
                    else:
                        shrunk = (n_e * own + N0 * pooled_mean) / (n_e + N0)
                        factor, meta = _factor_from_sigma(shrunk, rank); fallback[variant]["prior"] += 1; source = "shrinkage"
                factor.save(factor_root(variant) / f"layer_{layer:02d}_expert_{e:03d}.pt",
                            {"layer": layer, "expert": e, "rank": rank, "routed_tokens": n_e, "source": source, **meta})
                if variant == MAIN_VARIANT:
                    f_own = _factor_from_sigma(own, rank)[1]["f_at_rank"] if n_e >= rank + 1 else float("nan")
                    f_rows.append({"layer": layer, "expert": e, "routed_tokens": n_e, "source": source,
                                   "f_at_k6": meta["f_at_rank"], "f_at_k6_own_sigma": f_own})
                    rows = collector.rows.get((layer, e))
                    if rows and n_e >= 64:
                        held = torch.cat(rows)[: HELD_OUT_ROWS].float()
                        signs = torch.ones(dim)
                        nar = factor.apply(held, signs).reshape(held.shape[0], -1, GROUP)
                        had = act.full_hadamard_rows(held, signs).reshape(held.shape[0], -1, GROUP)
                        range_nar = float((nar.amax(-1) - nar.amin(-1)).mean())
                        range_had = float((had.amax(-1) - had.amin(-1)).mean())
                        law_rows.append({"layer": layer, "expert": e, "routed_tokens": n_e, "rows": held.shape[0],
                                         "f_at_k6": meta["f_at_rank"], "range_nar_over_hadamard": range_nar / range_had,
                                         "predicted_sqrt_1_minus_f": math.sqrt(max(0.0, 1 - meta["f_at_rank"]))})
        LOG.info("E26 factors layer %d/%d: pooled f=%.3f, cold experts (<%d tokens) %d/%d", layer + 1, layers,
                 pooled_meta["f_at_rank"], N0, int((counts < N0).sum()), experts)
    base.write_csv(result_dir() / f"{PREFIX}_expert_routing_counts.csv", count_rows)
    base.write_csv(result_dir() / f"{PREFIX}_expert_f.csv", f_rows)
    base.write_csv(result_dir() / f"{PREFIX}_expert_range_law.csv", law_rows)
    f_by_layer = []
    for layer in range(layers):
        vals = np.array([r["f_at_k6"] for r in f_rows if r["layer"] == layer])
        f_by_layer.append({"layer": layer, "median": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
                           "p90": float(np.percentile(vals, 90))})
    for variant in VARIANTS:
        base.atomic_json(factor_root(variant) / "DONE.json", {
            "model": MODEL_KEY, "variant": variant, "rank": rank, "n0": N0, "layers": layers, "experts": experts,
            "calibration_sequences": args.calibration_sequences, "sequence_length": args.seq_len,
            "fallback_counts": fallback[variant], "f_at_k6_by_layer": f_by_layer if variant == MAIN_VARIANT else None})
    del collector, model
    gc.collect(); torch.cuda.empty_cache()


def collector_total(counts: torch.Tensor) -> int:
    return int(counts.sum()) // 8 if int(counts.sum()) else 0  # tokens, not slots (top-8)


# ---------------------------------------------------------------- control ---

class RoutingRecorder:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.indices: dict[int, list[torch.Tensor]] = {}
        self.gaps: dict[int, list[torch.Tensor]] = {}
        self.handles: list[Any] = []

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers):
            def hook(_m: torch.nn.Module, _i: tuple, output: tuple, layer: int = layer) -> None:
                self.indices.setdefault(layer, []).append(output[2].detach().to(torch.int16).cpu())
                # Margin between the 8th and 9th router logit: a flip whose
                # reference margin sits at the fp32 noise floor is a tie, not
                # a routing change (softmax and top-k are monotone in the logit).
                top9 = output[0].detach().float().reshape(-1, output[0].shape[-1]).topk(9, dim=-1).values
                self.gaps.setdefault(layer, []).append((top9[:, 7] - top9[:, 8]).cpu())
            self.handles.append(block.mlp.gate.register_forward_hook(hook))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.inference_mode()
def _nll_and_routing(model: torch.nn.Module, tokens: torch.Tensor, label: str
                     ) -> tuple[list[float], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    recorder = RoutingRecorder(model)
    recorder.install()
    nll: list[float] = []
    try:
        device = model.get_input_embeddings().weight.device
        for index in range(tokens.shape[0]):
            batch = tokens[index:index + 1].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
            nll.append(float(loss))
            if index % 16 == 0:
                LOG.info("%s chunk %d/%d nll=%.6f", label, index + 1, tokens.shape[0], float(loss))
    finally:
        recorder.close()
    return (nll, {layer: torch.cat(v) for layer, v in recorder.indices.items()},
            {layer: torch.cat(v) for layer, v in recorder.gaps.items()})


def routing_agreement(reference: dict[int, torch.Tensor], candidate: dict[int, torch.Tensor],
                      reference_gaps: dict[int, torch.Tensor] | None = None, tie_gap: float = 1e-3) -> list[dict[str, Any]]:
    """Per-layer top-8 / argmax agreement, raw and with reference ties set aside.

    A token whose reference 8th-vs-9th logit margin is below `tie_gap` has no
    well-defined routing at fp32 precision: any reordering of the arithmetic
    (the fold is exact only in exact arithmetic) can flip it.  The raw
    agreement is reported as measured; the non-tie agreement is the fraction
    of tokens with a real margin whose routing survived the fold, and the
    largest reference margin among the flipped tokens says how far from a
    tie any flip was.
    """
    rows = []
    for layer in sorted(reference):
        a, b = reference[layer].long(), candidate[layer].long()
        set_same = (torch.sort(a, dim=-1).values == torch.sort(b, dim=-1).values).all(-1)
        argmax_same = a[:, 0] == b[:, 0]
        flips = (~set_same).nonzero().flatten().tolist()
        row = {"layer": layer, "tokens": int(a.shape[0]), "top8_set_agreement": float(set_same.float().mean()),
               "argmax_agreement": float(argmax_same.float().mean()), "set_flips": len(flips),
               "flip_positions": flips[:20]}
        if reference_gaps is not None and layer in reference_gaps:
            gap = reference_gaps[layer].float()
            real = gap > tie_gap
            row.update({
                "tie_gap": tie_gap, "tie_share": float((~real).float().mean()),
                "median_gap": float(gap.median()),
                "nontie_top8_agreement": float(set_same[real].float().mean()) if bool(real.any()) else 1.0,
                "nontie_argmax_agreement": float(argmax_same[real].float().mean()) if bool(real.any()) else 1.0,
                "max_flip_gap": float(gap[~set_same].max()) if flips else 0.0,
                "flips_above_tie_gap": int((~set_same & real).sum()),
            })
        rows.append(row)
    return rows


def routing_verdict(max_ratio: float = 2.0, min_floor_flips: int = 20) -> dict[str, Any]:
    """Judge the routing audit against the fp32 floor the null probe measured.

    A 48-layer top-8 router cannot be reproduced token for token in fp32.  The
    null probe applies R4_e and its transpose at the expert input and leaves
    every weight alone, so it changes nothing mathematically, and it still
    moves the top-8 set for 2,040 of 6.29M token-layer decisions, all of them
    deep, where the 8th-vs-9th logit margin has a median of 0.07.  An absolute
    per-layer threshold therefore measures fp32, not the fold.

    The verdict is the aggregate: how much routing the fold moves relative to
    how much that identity moves.  Per-layer ratios are reported only for
    layers where the floor itself has enough flips to make a ratio meaningful
    (a layer where the floor flips one token and the fold flips seven is two
    ten-thousandths of a percent either way).
    """
    directory = result_dir()
    floor_rows = {int(r["layer"]): r for r in base.read_csv(directory / f"{PREFIX}_null_routing_agreement.csv")}
    rows = base.read_csv(directory / f"{PREFIX}_routing_agreement.csv")
    control = json.loads((directory / f"{PREFIX}_control.json").read_text())
    floor_total = sum(int(r["set_flips"]) for r in floor_rows.values())
    decisions = sum(int(r["tokens"]) for r in floor_rows.values())
    verdicts: list[dict[str, Any]] = []
    for row in control["rows"]:
        rotation = row["rotation"]
        layers = [r for r in rows if r["rotation"] == rotation]
        total = sum(int(r["set_flips"]) for r in layers)
        ratios = [(int(r["layer"]), int(r["set_flips"]) / int(floor_rows[int(r["layer"])]["set_flips"]))
                  for r in layers
                  if int(floor_rows.get(int(r["layer"]), {"set_flips": 0})["set_flips"]) >= min_floor_flips]
        worst_layer, worst_ratio = max(ratios, key=lambda item: item[1], default=(None, 0.0))
        ratio = total / floor_total if floor_total else float("inf")
        verdicts.append({
            "rotation": rotation, "ppl_abs_difference": row["ppl_abs_difference"],
            "top8_flips": total, "floor_top8_flips": floor_total, "token_layer_decisions": decisions,
            "flip_rate": total / decisions, "floor_flip_rate": floor_total / decisions,
            "disagreement_ratio_to_floor": ratio,
            "worst_deep_layer_ratio": worst_ratio, "worst_deep_layer": worst_layer,
            "min_layer_top8_agreement": row["min_layer_top8_agreement"],
            "floor_min_layer_top8_agreement": min(float(r["top8_set_agreement"]) for r in floor_rows.values()),
            "max_ratio": max_ratio,
            "passed": bool(row["ppl_abs_difference"] <= control["ppl_tolerance"] and ratio <= max_ratio)})
    payload = {"model": MODEL_KEY, "criterion":
               "PPL within tolerance and aggregate top-8 routing disagreement at most "
               f"{max_ratio}x what an exact identity reordering produces (the null probe). "
               "The absolute 0.999 per-layer threshold is below the fp32 floor and is reported, not applied.",
               "floor_source": f"{PREFIX}_null_control.json", "rows": verdicts,
               "git_commit": e19.git_commit()}
    base.atomic_json(directory / f"{PREFIX}_control_verdict.json", payload)
    return payload


def verdict_command(args: argparse.Namespace) -> None:
    payload = routing_verdict(args.max_ratio)
    for row in payload["rows"]:
        print(f"{row['rotation']:10s} ppl_diff={row['ppl_abs_difference']:.2e} "
              f"flips={row['top8_flips']} vs floor {row['floor_top8_flips']} "
              f"({row['disagreement_ratio_to_floor']:.2f}x) worst deep layer {row['worst_deep_layer']} "
              f"{row['worst_deep_layer_ratio']:.2f}x  min_agree={row['min_layer_top8_agreement']:.6f} "
              f"(floor {row['floor_min_layer_top8_agreement']:.6f})  passed={row['passed']}")


def control_command(args: argparse.Namespace) -> None:
    """Rotation-only control (fold, no quantizer) and the routing-agreement audit."""
    base.setup_logging(WORKDIR, "e26-control")
    tokens = base.prepare_token_chunks(MODEL_ID, "test", 0, args.control_chunks, args.seq_len, WORKDIR)
    ref_path = result_dir() / f"{PREFIX}_control_reference.pt"
    payload = torch.load(ref_path, weights_only=True) if ref_path.exists() else None
    if payload is not None and "gaps" in payload:
        ref_nll, ref_routing, ref_gaps = payload["nll"], payload["routing"], payload["gaps"]
    else:
        model = load_sharded()
        ref_nll, ref_routing, ref_gaps = _nll_and_routing(model, tokens, "E26 fp32 reference")
        base.atomic_torch_save(ref_path, {"nll": ref_nll, "routing": ref_routing, "gaps": ref_gaps})
        del model; gc.collect(); torch.cuda.empty_cache()
    out_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    for rotation in args.rotations or ROTATIONS:
        # "null" is not a rotation: it applies R4_e and its transpose at the
        # expert input with the weights left alone.  In exact arithmetic it
        # changes nothing, so whatever routing it moves is the fp32 floor the
        # real rotations cannot beat.
        null = rotation == "null"
        model = load_sharded()
        rotations = MoERotationSet(WORKDIR, MODEL_KEY, "nar_kmax" if null else rotation, args.seed,
                                   model.config, torch.device("cuda:0"), MAIN_VARIANT)
        e14.FOLD_DEVICE = torch.device("cuda:0")
        if not null:
            fuse_norms_and_rotate_moe(model, rotations, args.weight_row_batch)
        hooks = MoERuntimeHooks(model, rotations, activation_kind=None, quantize_kv=False, round_trip=null)
        hooks.install()
        try:
            nll, routing, _ = _nll_and_routing(model, tokens, f"E26 rotation-only {rotation}")
        finally:
            hooks.close()
        deltas = np.array(nll) - np.array(ref_nll)
        agreement = routing_agreement(ref_routing, routing, ref_gaps, args.routing_tie_gap)
        for r in agreement:
            routing_rows.append({"rotation": rotation, **r})
        worst = min(r["top8_set_agreement"] for r in agreement)
        worst_nontie = min(r["nontie_top8_agreement"] for r in agreement)
        ppl_diff = abs(math.exp(float(np.mean(nll))) - math.exp(float(np.mean(ref_nll))))
        row = {"rotation": rotation, "chunks": len(nll), "reference_ppl": math.exp(float(np.mean(ref_nll))),
               "rotation_only_ppl": math.exp(float(np.mean(nll))), "mean_nll_delta": float(deltas.mean()),
               "max_abs_nll_delta": float(np.abs(deltas).max()), "ppl_abs_difference": ppl_diff,
               "min_layer_top8_agreement": worst, "min_layer_argmax_agreement": min(r["argmax_agreement"] for r in agreement),
               "mean_layer_top8_agreement": float(np.mean([r["top8_set_agreement"] for r in agreement])),
               "tie_gap": args.routing_tie_gap, "max_tie_share": max(r["tie_share"] for r in agreement),
               "min_layer_nontie_top8_agreement": worst_nontie,
               "min_layer_nontie_argmax_agreement": min(r["nontie_argmax_agreement"] for r in agreement),
               "max_flip_gap": max(r["max_flip_gap"] for r in agreement),
               "flips_above_tie_gap": sum(r["flips_above_tie_gap"] for r in agreement),
               "raw_gate_passed": bool(ppl_diff <= args.control_ppl_tolerance and worst >= args.routing_tolerance),
               "passed": bool(ppl_diff <= args.control_ppl_tolerance and worst_nontie >= args.routing_tolerance)}
        out_rows.append(row)
        LOG.info("E26 control %s: %s", rotation, json.dumps(row))
        del model, hooks, rotations; gc.collect(); torch.cuda.empty_cache()
    # The null probe is a measurement of the floor, not a gate: it keeps its
    # own files so the real control's evidence is never overwritten.
    stem = f"{PREFIX}_null" if list(args.rotations or ROTATIONS) == ["null"] else PREFIX
    base.write_csv(result_dir() / f"{stem}_rotation_only_control.csv", out_rows)
    base.write_csv(result_dir() / f"{stem}_routing_agreement.csv", routing_rows)
    base.atomic_json(result_dir() / f"{stem}_control.json", {
        "model": MODEL_KEY, "chunks": args.control_chunks, "rows": out_rows, "ppl_tolerance": args.control_ppl_tolerance,
        "routing_tolerance": args.routing_tolerance, "routing_tie_gap": args.routing_tie_gap,
        "gate": "ppl within tolerance and every layer's top-8 agreement among non-tie tokens (reference margin > tie gap) "
                "at or above routing_tolerance; raw agreement recorded as measured",
        "git_commit": e19.git_commit(), "hardware": base.hardware_info()})
    failed = [r["rotation"] for r in out_rows if not r["passed"] and r["rotation"] != "null"]
    if failed:
        raise AssertionError(f"E26 rotation-only control failed for {failed}")


# ------------------------------------------------------------------- GPTQ ---

def _attention_groups(layer: torch.nn.Module) -> list[list[tuple[str, torch.nn.Linear]]]:
    return [[("self_attn.k_proj", layer.self_attn.k_proj), ("self_attn.v_proj", layer.self_attn.v_proj),
             ("self_attn.q_proj", layer.self_attn.q_proj)], [("self_attn.o_proj", layer.self_attn.o_proj)]]


def _layer_state(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {name: m.weight.detach().cpu() for group in _attention_groups(layer) for name, m in group}
    state["mlp.experts.gate_up_proj"] = layer.mlp.experts.gate_up_proj.detach().cpu()
    state["mlp.experts.down_proj"] = layer.mlp.experts.down_proj.detach().cpu()
    return state


def _load_layer_state(layer: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    modules = {name: m for group in _attention_groups(layer) for name, m in group}
    for name, m in modules.items():
        m.weight.data.copy_(payload[name].to(m.weight.dtype))
    layer.mlp.experts.gate_up_proj.data.copy_(payload["mlp.experts.gate_up_proj"].to(layer.mlp.experts.gate_up_proj.dtype))
    layer.mlp.experts.down_proj.data.copy_(payload["mlp.experts.down_proj"].to(layer.mlp.experts.down_proj.dtype))


def _slice_linear(param: torch.Tensor, e: int) -> torch.nn.Linear:
    """An nn.Linear whose weight IS expert e's slice (shared storage) for the GPTQ engine."""
    view = param.data[e]
    lin = torch.nn.Linear(view.shape[1], view.shape[0], bias=False, device="meta")
    lin.weight = torch.nn.Parameter(view, requires_grad=False)
    return lin


def _rotation_matrix(rotations: MoERotationSet, layer: int, e: int, n: int, device: torch.device) -> torch.Tensor:
    """G such that apply_expert(x) = x @ G^T (rows), by applying it to the identity."""
    return rotations.apply_expert(layer, e, torch.eye(n, device=device, dtype=torch.float32))


def gptq_command(args: argparse.Namespace) -> None:
    variant = args.variant
    output = checkpoint_dir(args.rotation, args.seed, variant)
    done = output / "DONE.json"
    if done.exists():
        LOG.info("E26 GPTQ checkpoint exists: %s", done)
        return
    output.mkdir(parents=True, exist_ok=True)
    quarot_gptq.FAILURE_DUMP_DIR = output
    base.setup_logging(WORKDIR, f"e26-gptq-{args.rotation}-{variant}")
    base.seed_everything(args.seed)
    settings = e14.WEIGHT_PROTOCOLS[PROTOCOL]
    tokens = e14._quarot_calibration_tokens(MODEL_ID, WORKDIR, args.calibration_sequences, args.seq_len, args.calibration_seed)
    # Fold drift probe as E14: reference logits from the same fp32 model
    # before it is folded in place (one 122 GB CPU model, not two).
    model = load_cpu()
    probe = tokens[:1, :args.verify_tokens]
    with torch.inference_mode():
        reference = model(input_ids=probe, use_cache=False).logits.float().clone()
    rotations = MoERotationSet(WORKDIR, MODEL_KEY, args.rotation, args.seed, model.config, torch.device("cuda:0"), variant)
    e14.FOLD_DEVICE = torch.device("cuda:0")
    fold = fuse_norms_and_rotate_moe(model, rotations, args.weight_row_batch)
    hooks = MoERuntimeHooks(model, rotations, activation_kind=None, quantize_kv=False)
    hooks.install()
    with torch.inference_mode():
        observed = model(input_ids=probe, use_cache=False).logits.float()
    invariance = {"max_abs_logit_error": float((observed - reference).abs().max()),
                  "relative_l2_logit_error": float((observed - reference).norm() / reference.norm().clamp_min(1e-30)),
                  "probe_tokens": int(probe.numel())}
    del observed, reference
    LOG.info("E26 fold relative logit error %.3g", invariance["relative_l2_logit_error"])

    rotary = model.model.rotary_emb.cuda()
    layers = model.model.layers
    stream_dtype = model.model.embed_tokens.weight.dtype
    hidden = torch.empty((tokens.shape[0], args.seq_len, model.config.hidden_size), dtype=stream_dtype)
    with torch.inference_mode():
        for index in range(tokens.shape[0]):
            hidden[index] = model.model.embed_tokens(tokens[index:index + 1]).squeeze(0)
    scratch = torch.empty_like(hidden)
    position_ids = torch.arange(args.seq_len, device="cuda").unsqueeze(0)
    dummy = torch.zeros((1, args.seq_len, model.config.hidden_size), device="cuda", dtype=stream_dtype)
    position_embeddings = rotary(dummy, position_ids)
    del dummy
    audit_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    started = time.time()
    n_experts = int(model.config.num_experts)

    def run_layer(layer: torch.nn.Module) -> None:
        with torch.inference_mode():
            for sequence in range(tokens.shape[0]):
                e14._layer_forward(layer, hidden[sequence:sequence + 1].cuda(), position_ids, position_embeddings)

    for layer_index, layer in enumerate(layers):
        layer_path = output / f"layer_{layer_index:02d}.pt"
        layer.cuda()
        if layer_path.exists():
            _load_layer_state(layer, layer_path)
        else:
            # Attention groups exactly as E14.
            for group_index, group in enumerate(_attention_groups(layer)):
                engines = {name: quarot_gptq.GPTQ(m, sym=settings["weight_sym"]) for name, m in group}
                handles = [m.register_forward_hook(lambda _m, inp, _o, name=name: engines[name].add_batch(inp[0].detach()))
                           for name, m in group]
                run_layer(layer)
                for h in handles:
                    h.remove()
                for name, _m in group:
                    audit = engines[name].fasterquant(blocksize=128, percdamp=0.01, groupsize=settings["weight_groupsize"],
                                                      act_order=settings["act_order"])
                    audit_rows.append({"layer": layer_index, "group": group_index, "module": name, **audit.__dict__})
                del engines; gc.collect(); torch.cuda.empty_cache()
            experts = layer.mlp.experts
            ctx = experts._nar_ctx
            # gate_up per expert: own routed rows, pooled fallback below GPTQ_MIN.
            gate_up_engines = {e: quarot_gptq.GPTQ(_slice_linear(experts.gate_up_proj, e), sym=settings["weight_sym"]) for e in range(n_experts)}
            pooled_gate = quarot_gptq.GPTQ(_slice_linear(experts.gate_up_proj, 0), sym=settings["weight_sym"])
            routed = torch.zeros(n_experts, dtype=torch.long)

            def gate_capture(e: int, x: torch.Tensor) -> None:
                gate_up_engines[e].add_batch(x.detach()); routed[e] += x.shape[0]

            ctx.gate_up_capture = gate_capture
            ctx.pooled_capture = lambda x: pooled_gate.add_batch(x.detach())
            run_layer(layer)
            ctx.gate_up_capture = None; ctx.pooled_capture = None
            for e in range(n_experts):
                engine = gate_up_engines[e]
                fallback = int(routed[e]) < GPTQ_MIN
                if fallback:
                    engine.hessian = pooled_gate.hessian.clone(); engine.nsamples = pooled_gate.nsamples
                before = experts.gate_up_proj.data[e].float().clone()
                audit = engine.fasterquant(blocksize=128, percdamp=0.01, groupsize=settings["weight_groupsize"], act_order=settings["act_order"])
                after = experts.gate_up_proj.data[e].float()
                expert_rows.append({"layer": layer_index, "expert": e, "module": "gate_up", "routed_tokens": int(routed[e]),
                                    "pooled_hessian": fallback, "relative_weight_error": float((after - before).norm() / before.norm()),
                                    "hessian_diag_max_over_median": audit.hessian_diag_max_over_median})
                del before
            del gate_up_engines, pooled_gate; gc.collect(); torch.cuda.empty_cache()
            # down per expert on the rotated 768-dim input; pooled Sigma rotated into the expert's basis.
            down_engines = {e: quarot_gptq.GPTQ(_slice_linear(experts.down_proj, e), sym=settings["weight_sym"]) for e in range(n_experts)}
            pooled_sigma = torch.zeros((experts.down_proj.shape[2], experts.down_proj.shape[2]), device="cuda", dtype=torch.float64)
            pooled_n = 0
            unrotated: dict[str, Any] = {"pooled_sigma": pooled_sigma, "n": 0}

            def down_capture(e: int, h: torch.Tensor) -> None:
                down_engines[e].add_batch(h.detach())

            def down_fn_capture(e: int, h: torch.Tensor, layer_index: int = layer_index) -> torch.Tensor:
                rows = h.detach().double()
                unrotated["pooled_sigma"] += rows.T @ rows; unrotated["n"] += rows.shape[0]
                return rotations.apply_expert(layer_index, e, h).to(h.dtype)

            saved_down_fn = ctx.down_fn
            ctx.down_fn = down_fn_capture; ctx.down_capture = down_capture
            run_layer(layer)
            ctx.down_fn = saved_down_fn; ctx.down_capture = None
            for e in range(n_experts):
                engine = down_engines[e]
                fallback = int(routed[e]) < GPTQ_MIN
                if fallback:
                    g = _rotation_matrix(rotations, layer_index, e, pooled_sigma.shape[0], pooled_sigma.device).double()
                    # add_batch's normalisation: hessian = 2/N * X^T X with N = sequences; pooled over tokens here.
                    engine.hessian = (2.0 * (g @ unrotated["pooled_sigma"] @ g.T) / max(unrotated["n"], 1)).to(engine.hessian_dtype)
                    engine.nsamples = max(1, unrotated["n"] // args.seq_len)
                before = experts.down_proj.data[e].float().clone()
                audit = engine.fasterquant(blocksize=128, percdamp=0.01, groupsize=settings["weight_groupsize"], act_order=settings["act_order"])
                after = experts.down_proj.data[e].float()
                expert_rows.append({"layer": layer_index, "expert": e, "module": "down", "routed_tokens": int(routed[e]),
                                    "pooled_hessian": fallback, "relative_weight_error": float((after - before).norm() / before.norm()),
                                    "hessian_diag_max_over_median": audit.hessian_diag_max_over_median})
                del before
            del down_engines; gc.collect(); torch.cuda.empty_cache()
        # Propagate through the (quantized) layer.
        with torch.inference_mode():
            for sequence in range(tokens.shape[0]):
                value = e14._layer_forward(layer, hidden[sequence:sequence + 1].cuda(), position_ids, position_embeddings)
                scratch[sequence].copy_(value.squeeze(0).cpu())
        layer.cpu()
        if not layer_path.exists():
            base.atomic_torch_save(layer_path, _layer_state(layer))
            base.write_csv(output / "gptq_audit.partial.csv", audit_rows)
            base.write_csv(output / "gptq_expert_audit.partial.csv", expert_rows)
        hidden, scratch = scratch, hidden
        LOG.info("E26 GPTQ layer %d/%d done (%.1f min elapsed)", layer_index + 1, len(layers), (time.time() - started) / 60)
        gc.collect(); torch.cuda.empty_cache()
    hooks.close()
    base.write_csv(output / "gptq_audit.csv", audit_rows)
    base.write_csv(output / "gptq_expert_audit.csv", expert_rows)
    cold = [r for r in expert_rows if r["pooled_hessian"]]
    base.atomic_json(done, {
        "model": MODEL_KEY, "model_id": MODEL_ID, "rotation": args.rotation, "variant": variant,
        "gptq": {**{k: v for k, v in settings.items()}, "protocol": PROTOCOL, "bits": 4, "blocksize": 128, "percdamp": 0.01,
                 "calibration_sequences": args.calibration_sequences, "calibration_seed": args.calibration_seed,
                 "sequence_length": args.seq_len, "per_expert_hessian": True, "pooled_hessian_below": GPTQ_MIN,
                 "experts_with_pooled_hessian": len({(r["layer"], r["expert"]) for r in cold}),
                 "excluded": ["embed_tokens", "lm_head", "router gates (bf16)"]},
        "fold": fold, "bf16_reparameterization_drift": invariance, "ranks": rotations.ranks(),
        "elapsed_seconds": time.time() - started, "hardware": base.hardware_info(), "git_commit": e19.git_commit()})
    (output / "gptq_audit.partial.csv").unlink(missing_ok=True)
    (output / "gptq_expert_audit.partial.csv").unlink(missing_ok=True)


# ------------------------------------------------------------- evaluation ---

def effective_bits(model_config: Any, quantized: bool, context: int) -> dict[str, Any]:
    h, kv, hd = int(model_config.hidden_size), int(model_config.num_key_value_heads) * 128, 128
    heads = int(model_config.num_attention_heads)
    experts, inter = int(model_config.num_experts), int(model_config.moe_intermediate_size)
    attn_values = h * heads * hd + 2 * h * kv + heads * hd * h
    attn_scales = heads * hd + 2 * kv + h
    expert_values = experts * (2 * inter * h + h * inter)
    expert_scales = experts * (2 * inter + h)
    router_values = experts * h
    w_bits = 4 + (16 + 4) * (attn_scales + expert_scales) / (attn_values + expert_values) if quantized else 16.0
    w_bits_with_router = ((w_bits * (attn_values + expert_values) + 16 * router_values) / (attn_values + expert_values + router_values)
                          if quantized else 16.0)
    bits = e19.effective_bits(ACTIVATION_KIND if quantized else None, quantized, quantized,
                              {"head_dim": hd, "hidden_size": h, "intermediate_size": inter, "num_key_value_heads": int(model_config.num_key_value_heads)},
                              context)
    bits["weight"] = w_bits
    bits["weight_including_bf16_router"] = w_bits_with_router
    bits["weight_definition"] = "GPTQ g128_asym over attention and all 128 experts per layer (fp16 scale + int4 zero per group); router gates bf16"
    return bits


def build_row(args: argparse.Namespace, row: str) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    rotation = ROWS[row]
    provenance: dict[str, Any] = {"model": MODEL_KEY, "model_id": MODEL_ID, "row": row, "rotation_checkpoint": rotation,
                                  "variant": args.variant, "seed": args.seed, "compute_dtype": "float32", "fold_dtype": "float32",
                                  "gptq_protocol": PROTOCOL if rotation else None, "activation_kind": ACTIVATION_KIND if rotation else None,
                                  "kv_policy": "E14 KIVI" if rotation else None, "git_commit": e19.git_commit(),
                                  "hardware": base.hardware_info()}
    model = load_sharded()
    provenance["effective_bits"] = effective_bits(model.config, bool(rotation), args.seq_len)
    if rotation is None:
        return e22._clear_generation_limits(model), None, provenance
    root = checkpoint_dir(rotation, args.seed, args.variant)
    if not (root / "DONE.json").exists():
        raise FileNotFoundError(root / "DONE.json")
    rotations = MoERotationSet(WORKDIR, MODEL_KEY, rotation, args.seed, model.config, torch.device("cuda:0"), args.variant)
    e14.FOLD_DEVICE = torch.device("cuda:0")
    fuse_norms_and_rotate_moe(model, rotations, args.weight_row_batch)
    for index, layer in enumerate(model.model.layers):
        _load_layer_state(layer, root / f"layer_{index:02d}.pt")
    done = json.loads((root / "DONE.json").read_text())
    provenance["gptq"] = done["gptq"]; provenance["ranks"] = rotations.ranks()
    hooks = MoERuntimeHooks(model, rotations, activation_kind=ACTIVATION_KIND, quantize_kv=True)
    hooks.install()
    return e22._clear_generation_limits(model), hooks, provenance


def evaluate_command(args: argparse.Namespace) -> None:
    suffix = "" if args.variant == MAIN_VARIANT else f"_{args.variant}"
    path = result_dir() / f"{PREFIX}_{args.row}{suffix}_{args.benchmark}.json"
    if path.exists():
        LOG.info("E26 row exists: %s", path)
        return
    base.setup_logging(WORKDIR, f"e26-evaluate-{args.row}{suffix}-{args.benchmark}")
    base.seed_everything(args.seed)
    started = time.time()
    model, hooks, provenance = build_row(args, args.row)
    try:
        spec = e22.harness_config(args.benchmark, MODEL_KEY) if args.benchmark in e22.BENCHMARKS else None
        if args.benchmark in ("wikitext", "c4"):
            tokens = (base.prepare_token_chunks(MODEL_ID, "test", 0, e19.EVAL_WINDOWS, args.seq_len, WORKDIR)
                      if args.benchmark == "wikitext" else e22.c4_tokens(MODEL_ID, args.seq_len, e22.C4_WINDOWS))
            ppl, rows = e19.evaluate_ppl_fp32(model, tokens, f"E26 {args.row} {args.benchmark}")
            payload = {**provenance, "benchmark": args.benchmark, "ppl": ppl, "chunks": rows, "chunks_evaluated": len(rows),
                       "sequence_length": args.seq_len, "nll_dtype": "float32", "headline_metric": "ppl", "headline": ppl}
        else:
            e19.MODEL_ID = MODEL_ID
            args.model = MODEL_KEY
            e22.DEFAULT_BATCH[MODEL_KEY] = 4
            result = e22.run_harness(model, args, spec, cache_key=f"{PREFIX}-{MODEL_KEY}-{args.row}{suffix}")
            metric_name, value = e22.headline(spec, result["results"])
            payload = {**provenance, "benchmark": args.benchmark, "tasks": spec["tasks"], "num_fewshot": spec.get("num_fewshot"),
                       "harness_commit": e14.HARNESS_COMMIT, "results": e14._serializable(result["results"]),
                       "sample_counts": e14._serializable(result.get("n-samples", {})), "headline_metric": metric_name, "headline": value}
        payload["elapsed_seconds"] = time.time() - started
        base.atomic_json(path, payload)
        LOG.info("E26 %s %s: %s = %s", args.row, args.benchmark, payload["headline_metric"], payload["headline"])
    finally:
        if hooks is not None:
            hooks.close()


def finalize_command(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted(result_dir().glob(f"{PREFIX}_*_*.json")):
        payload = json.loads(path.read_text())
        if "headline" in payload and "row" in payload:
            rows.append({"row": payload["row"], "variant": payload.get("variant"), "benchmark": payload["benchmark"],
                         "value": payload["headline"], "effective_bits": json.dumps(payload.get("effective_bits")),
                         "artifact": str(path.relative_to(WORKDIR))})
    base.write_csv(result_dir() / f"{PREFIX}_summary.csv", rows)
    for r in rows:
        print(f"{r['row']:22s} {str(r['variant']):10s} {r['benchmark']:10s} {r['value']:9.3f}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--weight-row-batch", type=int, default=256)
    p.add_argument("--calibration-sequences", type=int, default=128)
    p.add_argument("--calibration-seed", type=int, default=0)
    p.add_argument("--verify-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--variant", choices=VARIANTS, default=MAIN_VARIANT)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("calibrate")
    control = sub.add_parser("control")
    control.add_argument("--rotations", nargs="+", choices=tuple(ROTATIONS) + ("null",))
    control.add_argument("--control-chunks", type=int, default=64)
    control.add_argument("--control-ppl-tolerance", type=float, default=0.01)
    control.add_argument("--routing-tolerance", type=float, default=0.999)
    control.add_argument("--routing-tie-gap", type=float, default=1e-3)
    verdict = sub.add_parser("verdict")
    verdict.add_argument("--max-ratio", type=float, default=2.0)
    gptq = sub.add_parser("gptq")
    gptq.add_argument("--rotation", choices=ROTATIONS, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--row", choices=tuple(ROWS), required=True)
    evaluate.add_argument("--benchmark", choices=("wikitext", "c4", "eight_task", "six_task"), required=True)
    sub.add_parser("finalize")
    return p


def main() -> None:
    global WORKDIR
    args = parser().parse_args()
    WORKDIR = Path(args.workdir).resolve()
    e22.WORKDIR = WORKDIR
    e19.MODEL_KEY = MODEL_KEY; e19.MODEL_ID = MODEL_ID; e19.PREFIX = PREFIX
    install_experts_patch()
    {"audit": audit_command, "calibrate": calibrate_command, "control": control_command, "gptq": gptq_command,
     "evaluate": evaluate_command, "finalize": finalize_command, "verdict": verdict_command}[args.command](args)


if __name__ == "__main__":
    main()
