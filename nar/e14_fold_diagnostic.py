#!/usr/bin/env python3
"""Stagewise numerical audit for the E14 QuaRot reparameterization.

This is a diagnostic only: it never changes a pre-registered quantization row
or acceptance threshold.  It separates algebraic mistakes from rounding drift
introduced when an exactly equivalent rotation is materialized in bf16.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Callable

import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from .e14_w4a4kv4 import RotationSet
except ImportError:
    import activation_experiments as act
    import experiment as base
    from e14_w4a4kv4 import RotationSet


def _metric(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = observed.float() - reference.float()
    return {
        "max_abs_logit_error": float(difference.abs().max()),
        "relative_l2_logit_error": float(difference.norm() / reference.float().norm().clamp_min(1e-30)),
    }


def _load_model(model_id: str, workdir: Path, model_dtype: str) -> torch.nn.Module:
    model = base.load_model(model_id, workdir)
    if model_dtype == "fp32":
        model.float()
    return model


@torch.inference_mode()
def _fuse_norms(model: torch.nn.Module) -> None:
    """Released QuaRot fusion precision, without its non-invariant centering."""
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    if tied:
        old = model.lm_head
        head = torch.nn.Linear(old.in_features, old.out_features, bias=False,
                               device=old.weight.device, dtype=old.weight.dtype)
        head.weight.copy_(old.weight)
        model.lm_head = head
        model.config.tie_word_embeddings = False
    for block in model.model.layers:
        for norm, modules in (
            (block.input_layernorm, (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj)),
            (block.post_attention_layernorm, (block.mlp.gate_proj, block.mlp.up_proj)),
        ):
            scale = norm.weight.detach().double()
            for module in modules:
                module.weight.data.copy_((module.weight.detach().double() * scale.unsqueeze(0)).to(module.weight.dtype))
            norm.weight.fill_(1)
    scale = model.model.norm.weight.detach().double()
    model.lm_head.weight.data.copy_((model.lm_head.weight.detach().double() * scale.unsqueeze(0)).to(model.lm_head.weight.dtype))
    model.model.norm.weight.fill_(1)


@torch.inference_mode()
def _right(module: torch.nn.Linear, transform: Callable[[torch.Tensor], torch.Tensor], batch: int) -> None:
    weight = module.weight.detach()
    output = [transform(weight[start:start + batch].float()).to(weight.dtype)
              for start in range(0, weight.shape[0], batch)]
    module.weight.data.copy_(torch.cat(output, 0))


@torch.inference_mode()
def _left(module: torch.nn.Linear, transform: Callable[[torch.Tensor], torch.Tensor], batch: int) -> None:
    weight_t = module.weight.detach().T
    output = [transform(weight_t[start:start + batch].float()).to(weight_t.dtype)
              for start in range(0, weight_t.shape[0], batch)]
    module.weight.data.copy_(torch.cat(output, 0).T)


@torch.inference_mode()
def _apply_stages(model: torch.nn.Module, rotations: RotationSet, stages: set[str], batch: int) -> None:
    _fuse_norms(model)
    if "r1" in stages:
        r1 = lambda value: rotations.apply("r1", 0, value)
        _right(model.model.embed_tokens, r1, batch)
        _right(model.lm_head, r1, batch)
        for block in model.model.layers:
            for module in (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj,
                           block.mlp.gate_proj, block.mlp.up_proj):
                _right(module, r1, batch)
            for module in (block.self_attn.o_proj, block.mlp.down_proj):
                _left(module, r1, batch)
    heads = int(model.config.num_attention_heads)
    head_dim = rotations.head_dim
    for layer, block in enumerate(model.model.layers):
        if "r2" in stages:
            weight = block.self_attn.o_proj.weight.detach()
            shaped = weight.reshape(weight.shape[0], heads, head_dim)
            rotated = rotations.apply("r2", layer, shaped.reshape(-1, head_dim)).to(weight.dtype)
            weight.copy_(rotated.reshape_as(shaped).reshape_as(weight))
        if "r4" in stages:
            _right(block.mlp.down_proj,
                   lambda value, layer=layer: rotations.apply("r4", layer, value), batch)


class _OnlineStages:
    def __init__(self, model: torch.nn.Module, rotations: RotationSet, stages: set[str]):
        self.handles = []
        if "r2" in stages:
            for layer, block in enumerate(model.model.layers):
                def rotate_v(_module, _inputs, output, layer=layer):
                    shape = output.shape
                    return rotations.apply("r2", layer, output.reshape(-1, rotations.head_dim)).reshape(shape).to(output.dtype)
                self.handles.append(block.self_attn.v_proj.register_forward_hook(rotate_v))
        if "r4" in stages:
            for layer, block in enumerate(model.model.layers):
                def rotate_down(_module, inputs, layer=layer):
                    return (rotations.apply("r4", layer, inputs[0]).to(inputs[0].dtype),) + inputs[1:]
                self.handles.append(block.mlp.down_proj.register_forward_pre_hook(rotate_down))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E14 fold diagnostic requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, f"e14-fold-diagnostic-{args.model}-{args.rotation}")
    model_id, model_key = act.model_id_and_key(args.model)
    tokens = base.prepare_token_chunks(model_id, "train", 0, 1, args.seq_len, workdir)
    probe = tokens[:, :args.tokens].cuda()
    original = _load_model(model_id, workdir, args.model_dtype)
    reference = original(input_ids=probe, use_cache=False).logits.float()
    del original
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for label, stages in (
        ("norm_only", set()),
        ("norm_r1", {"r1"}),
        ("norm_r1_r2", {"r1", "r2"}),
        ("norm_r1_r4", {"r1", "r4"}),
        ("norm_r1_r2_r4", {"r1", "r2", "r4"}),
    ):
        model = _load_model(model_id, workdir, args.model_dtype)
        rotations = RotationSet(workdir, model_key, args.rotation, args.seed,
                                model.config, torch.device("cuda"))
        _apply_stages(model, rotations, stages, args.weight_row_batch)
        online = _OnlineStages(model, rotations, stages)
        try:
            observed = model(input_ids=probe, use_cache=False).logits.float()
        finally:
            online.close()
        rows.append({"model": model_key, "rotation": args.rotation, "stage": label,
                     "model_dtype": str(model.dtype), "tokens": int(probe.numel()),
                     **_metric(observed, reference)})
        del model, rotations, observed
        gc.collect()
        torch.cuda.empty_cache()
    output = workdir / "results" / model_key / f"e14_{args.rotation}_{args.model_dtype}_fold_diagnostic.csv"
    base.write_csv(output, rows)
    base.atomic_json(output.with_suffix(".json"), {"rows": rows, "diagnostic_only": True,
                                                   "embedding_centering": False,
                                                   "seed": args.seed,
                                                   "hardware": base.hardware_info()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--model", choices=("llama32_3b", "llama31_8b"), default="llama32_3b")
    parser.add_argument("--rotation", choices=("hadamard", "nar"), default="hadamard")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--weight-row-batch", type=int, default=256)
    parser.add_argument("--model-dtype", choices=("bf16", "fp32"), default="bf16")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
