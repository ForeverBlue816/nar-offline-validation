#!/usr/bin/env python3
"""QuaRot's e2e Llama model classes, ported to Llama-3 (E28).

This file is ``QuaRot/e2e/quantized_llama/modeling_llama.py`` at commit 5008669
with the minimum changes needed for Llama-3.2-3B / Llama-3.1-8B.  It runs on
transformers 4.36.2, the version QuaRot's e2e code was written against (the
repository's last commit is a dependabot bump of the pin to 4.38.0, under which
the released e2e model does not run: 4.38.0 routes every non-static cache
through ``DynamicCache.from_legacy_cache`` and changed the rotary-embedding call
signature).  Every change is marked ``# E28:`` and
listed in ``quarot-llama3/README.md``.  Three model classes:

  QuarotFP16LlamaForCausalLM   row 1: fp16 weights/activations, fp16 paged KV
                               (QuaRot's own baseline: HF Llama + flash-attn
                               prefill + QuaRot FlashInfer fp16 decode)
  QuarotLlamaForCausalLM       row 2: QuaRot INT4 (CUTLASS GEMM, INT4 KV,
                               fused Hadamard R4)
  QuarotNARLlamaForCausalLM    row 3: identical to row 2 except the R4 slot,
                               where OnlineHadamard is replaced by the E17 v3
                               NAR kernel (nar_r4_kernel.NARDownTransform)
"""

from __future__ import annotations

import json
import math
from typing import Optional, Tuple

import quarot
import quarot.transformers
import torch
from transformers import Cache, LlamaConfig
from transformers.models.llama.modeling_llama import (LlamaFlashAttention2, LlamaForCausalLM,
                                                      LlamaMLP, apply_rotary_pos_emb)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

ALL_LAYERNORM_LAYERS.append(quarot.nn.RMSNorm)


# E28: transformers 4.36 rejects Llama-3's ``rope_scaling: {"rope_type": "llama3", ...}``
# (it only knows {"type": linear|dynamic, "factor"}), so the config is loaded with
# rope_scaling removed and the Llama-3 inverse frequencies are installed on every
# layer's rotary embedding afterwards (``install_llama3_rope``).  This reproduces
# transformers>=4.43 ``_compute_llama3_parameters`` exactly.
def load_llama3_config(config_path: str) -> LlamaConfig:
    payload = json.loads(open(config_path).read())
    rope_scaling = payload.pop("rope_scaling", None)
    payload.pop("transformers_version", None)
    config = LlamaConfig(**payload)
    config.llama3_rope_scaling = rope_scaling
    config._attn_implementation = "flash_attention_2"
    return config


def llama3_inv_freq(config: LlamaConfig, device: torch.device) -> torch.Tensor:
    head_dim = config.hidden_size // config.num_attention_heads
    base = float(config.rope_theta)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim))
    scaling = getattr(config, "llama3_rope_scaling", None)
    if not scaling:
        return inv_freq
    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_context_len = scaling["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def install_llama3_rope(model: LlamaForCausalLM) -> None:
    for layer in model.model.layers:
        rotary = layer.self_attn.rotary_emb
        rotary.inv_freq = llama3_inv_freq(model.config, rotary.inv_freq.device).to(rotary.inv_freq.dtype)


class QuarotLlamaConfig(LlamaConfig):
    model_type = "llama_quarot"


class OnlineHadamardDense(torch.nn.Module):
    """E28: QuaRot's OnlineHadamard for a non-power-of-two head count, as one dense matmul.

    ``quarot.nn.OnlineHadamard(num_heads)`` is applied over the head axis of the
    attention output.  For 24 heads (Llama-3.2-3B) QuaRot's ``matmul_hadU_cuda``
    factors 24 = 12 x 2 and runs fast-hadamard-transform on a last dimension of 2
    followed by a broadcast ``had_12 @ x`` over batch x tokens x head_dim = 3M
    tiny (12x12)@(12x2) products; on the A40 that costs 2.2 ms per layer at 2048
    tokens, 30% of the whole prefill.  Llama-2's 32 heads never hit this path
    (K = 1).  This module applies the identical transform H_24 = (had_12 kron H_2)
    / sqrt(24) as a single (tokens*head_dim, 24) @ (24, 24) fp16 GEMM.  The
    matrix is checked against ``matmul_hadU_cuda`` at construction.  Used by rows
    2 and 3 alike; power-of-two head counts (Llama-3.1-8B) keep QuaRot's kernel.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        from quarot.functional.hadamard import get_hadK, matmul_hadU_cuda
        had_k, k = get_hadK(num_heads)
        assert had_k is not None and k > 1
        h2 = torch.ones((1, 1))
        while h2.shape[0] < num_heads // k:
            h2 = torch.kron(torch.tensor([[1.0, 1.0], [1.0, -1.0]]), h2)
        dense = torch.kron(had_k.float(), h2) / math.sqrt(num_heads)   # y = dense @ x (columns)
        probe = torch.randn(64, num_heads, device="cuda", dtype=torch.float16)
        reference = matmul_hadU_cuda(probe, had_k.to(torch.float16), k).float()
        observed = probe.float() @ dense.cuda().T
        assert torch.allclose(observed, reference, atol=2e-3, rtol=2e-3), (observed - reference).abs().max()
        self.register_buffer("weight_t", dense.T.contiguous().to(torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight_t.to(x.dtype))


def online_head_hadamard(num_heads: int) -> torch.nn.Module:
    from quarot.functional.hadamard import get_hadK
    _, k = get_hadK(num_heads)
    return quarot.nn.OnlineHadamard(num_heads) if k == 1 else OnlineHadamardDense(num_heads)


class QuarotFP16LlamaAttention(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = torch.nn.Identity()
        self.o_proj_hadamard = torch.nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        kv_seq_len = key_states.shape[1]
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, unsqueeze_dim=2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        assert past_key_value is not None
        # sin and cos are specific to RoPE models; position_ids needed for the static cache

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "attention_mask": attention_mask}
        cache_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        dropout_rate = self.attention_dropout if self.training else 0.0

        assert self.is_causal

        if isinstance(cache_out, tuple):
            key_states, value_states = cache_out
            attn_output = self._flash_attention_forward(
                query_states,
                key_states,
                value_states,
                query_length=q_len,
                attention_mask=attention_mask
            )
        else:
            attn_output = cache_out(query_states)

        attn_output = self.o_proj_hadamard(attn_output.transpose(-1, -2)).transpose(-1, -2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuarotLlamaAttention(QuarotFP16LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot.nn.Quantizer()
        self.q_proj = quarot.nn.Linear4bit.from_float(self.q_proj)
        self.k_proj = quarot.nn.Linear4bit.from_float(self.k_proj)
        self.v_proj = quarot.nn.Linear4bit.from_float(self.v_proj)
        self.o_proj_hadamard = online_head_hadamard(self.num_heads)  # E28: see OnlineHadamardDense
        self.o_proj = torch.nn.Sequential(
            quarot.nn.Quantizer(),
            quarot.nn.Linear4bit.from_float(self.o_proj)
        )


class QuarotLlamaMLP(LlamaMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quarot.nn.Quantizer()
        self.up_proj = quarot.nn.Linear4bit.from_float(self.up_proj)
        self.gate_proj = quarot.nn.Linear4bit.from_float(self.gate_proj)
        self.down_proj = torch.nn.Sequential(
            quarot.nn.OnlineHadamard(self.intermediate_size),
            quarot.nn.Quantizer(),
            quarot.nn.Linear4bit.from_float(self.down_proj)
        )

    def forward(self, x):
        x = self.quantizer(x)
        return super().forward(x)


class QuarotNARLlamaMLP(QuarotLlamaMLP):
    """E28 row 3: the R4 slot holds the E17 v3 NAR kernel instead of OnlineHadamard."""

    def __init__(self, config, nar_transform: torch.nn.Module):
        super().__init__(config)
        self.down_proj = torch.nn.Sequential(
            nar_transform,
            self.down_proj[1],   # the same quarot.nn.Quantizer
            self.down_proj[2],   # the same quarot.nn.Linear4bit
        )


class QuarotFP16LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotFP16LlamaAttention(config=config, layer_idx=layer_idx)
        self.cache_dtype = "float16"
        self._expected_max_length = None
        # E28: serving-style prefill — logits only for the last position (see forward).
        self.logits_last_only = False

    def build_cache(self, batch_size, page_size, max_length):
        device = self.model.layers[0].self_attn.v_proj.weight.device
        dtype = self.cache_dtype or self.model.layers[0].self_attn.v_proj.weight.dtype

        num_heads = self.config.num_key_value_heads
        model_dim = self.config.hidden_size
        # E28: the released code divides hidden_size by num_key_value_heads, which is
        # only the head dimension when there is no GQA (Llama-2 7B/13B).  Llama-3 has
        # 8 KV heads for 24/32 query heads, so the head dimension comes from the
        # attention heads (3072/24 = 4096/32 = 128).
        head_dim = model_dim // self.config.num_attention_heads
        disable_quant = self.cache_dtype == "float16"
        return quarot.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size,
            page_size=page_size,
            max_seq_len=max_length,
            device=device,
            n_layers=len(self.model.layers),
            num_heads=num_heads,
            head_dim=head_dim,
            disable_quant=disable_quant,
            hadamard_dtype=None if disable_quant else torch.float16
        )

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        # This is a hack to get the max length from generation_config.
        # Doing it here because max_length might not be set before this
        # method is called.
        self._expected_max_length = generation_config.max_length # This value will be reset at the next forward call
        return super()._get_logits_processor(generation_config, *args, **kwargs)

    def forward(self, input_ids, *args, past_key_values=None, **kwargs):
        if past_key_values is None:
            max_length = self._expected_max_length or input_ids.shape[1]
            self._expected_max_length = None # Reset this value.
            past_key_values = self.build_cache(
                input_ids.shape[0],
                page_size=max_length,  # For now working with single page per batch.
                max_length=max_length)
        if not self.logits_last_only:
            return super().forward(input_ids, *args, past_key_values=past_key_values, **kwargs)
        # E28: LlamaForCausalLM.forward materialises fp16 logits for every position and
        # then upcasts them to fp32 (batch 16 x 2048 x 128256 vocab = 25 GB for Llama-3),
        # which does not fit next to the 8B fp16 weights on a 48 GB A40 and would
        # dominate the peak-memory metric of every row.  Prefill in serving computes
        # logits for the last position only; that is done here for all three rows.
        outputs = self.model(input_ids, *args, past_key_values=past_key_values, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -1:, :]).float()
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)


class QuarotLlamaForCausalLM(QuarotFP16LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        assert config._attn_implementation == "flash_attention_2"
        self.norm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = QuarotLlamaAttention(config=config, layer_idx=layer_idx)
            layer.input_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = quarot.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            layer.mlp = QuarotLlamaMLP(config=config)
        self.cache_dtype = "int4"


class QuarotNARLlamaForCausalLM(QuarotLlamaForCausalLM):
    """E28 row 3.  ``nar_transforms[i]`` is layer i's NARDownTransform."""

    def __init__(self, config, nar_transforms):
        super().__init__(config)
        assert len(nar_transforms) == len(self.model.layers)
        for layer, transform in zip(self.model.layers, nar_transforms):
            layer.mlp = QuarotNARLlamaMLP(config, transform)
