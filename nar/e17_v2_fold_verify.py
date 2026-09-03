#!/usr/bin/env python3
"""Real-model verification of the E17 v2 signed-permutation fold."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:
    from . import activation_experiments as act
    from . import experiment as base
    from .fold_signed_permutation import FoldedR4, fold_swiglu_weights
except ImportError:
    import activation_experiments as act
    import experiment as base
    from fold_signed_permutation import FoldedR4, fold_swiglu_weights


MODELS = {
    "llama32_3b": {"id": "unsloth/Llama-3.2-3B", "layers": 28},
    "llama31_8b": {"id": "unsloth/Meta-Llama-3.1-8B", "layers": 32},
}


def factor_path(workdir: Path, model: str, layer: int) -> Path:
    return (
        workdir / "activations" / model / "e11_calibration" / "factors"
        / "nar_b128_k8" / f"down_layer_{layer:02d}.pt"
    )


def signs_for(n: int, layer: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(act._seed(seed, 0, layer, "down"))
    return torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1).to(device)


class Capture:
    def __init__(self, model: torch.nn.Module, layers: list[int], rows: int):
        self.model = model
        self.layers = layers
        self.rows = rows
        self.hidden: dict[int, torch.Tensor] = {}
        self.down: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _save(self, destination: dict[int, torch.Tensor], layer: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[: self.rows]
            destination[layer] = value.to("cpu", torch.bfloat16).clone()
        return hook

    def install(self) -> None:
        for layer in self.layers:
            block = self.model.model.layers[layer]
            self.handles.append(block.mlp.gate_proj.register_forward_pre_hook(self._save(self.hidden, layer)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self._save(self.down, layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def verify_model(workdir: Path, model_key: str, seed: int, rows: int) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    spec = MODELS[model_key]
    model = base.load_model(spec["id"], workdir)
    device = next(model.parameters()).device
    layers = [0, int(spec["layers"]) // 2]
    tokens = base.prepare_token_chunks(spec["id"], "train", 0, 1, 2048, workdir)
    capture = Capture(model, layers, max(rows, 64))
    capture.install()
    try:
        with torch.inference_mode():
            model(input_ids=tokens.cuda(non_blocking=True), use_cache=False)
    finally:
        capture.close()

    results: list[dict[str, Any]] = []
    cached: dict[str, torch.Tensor] = {}
    for layer in layers:
        block = model.model.layers[layer]
        gate, up = block.mlp.gate_proj, block.mlp.up_proj
        if gate.bias is not None or up.bias is not None:
            raise AssertionError(f"{model_key} layer {layer} MLP projection has bias")
        factor = act.RotationFactor.load(factor_path(workdir, model_key, layer), device)
        signs = signs_for(factor.n, layer, seed, device)
        folded = FoldedR4.from_factor(factor, signs)
        hidden = capture.hidden[layer][:rows].to(device)
        down = capture.down[layer][:rows].to(device)
        source = folded.source.to(device)
        gate_new = gate.weight.detach().index_select(0, source)
        up_new = up.weight.detach().index_select(0, source) * signs.to(up.weight.dtype).unsqueeze(1)
        with torch.inference_mode():
            q_from_weights = F.silu(F.linear(hidden, gate_new)) * F.linear(hidden, up_new)
            q_reference = folded.q_unfolded(down)
            old_output = factor.apply(down.float(), signs)
            new_output = folded.apply(q_from_weights.float())
        difference = new_output - old_output
        row_relative = difference.norm(dim=-1) / old_output.norm(dim=-1).clamp_min(1e-30)
        results.append({
            "model": model_key,
            "layer": layer,
            "rows": rows,
            "gate_bias": gate.bias is not None,
            "up_bias": up.bias is not None,
            "qx_max_abs": float((q_from_weights.float() - q_reference).abs().max()),
            "qx_relative_l2": float((q_from_weights.float() - q_reference).norm() / q_reference.norm().clamp_min(1e-30)),
            "operator_max_abs": float(difference.abs().max()),
            "operator_relative_l2": float(difference.norm() / old_output.norm().clamp_min(1e-30)),
            "operator_max_row_relative_l2": float(row_relative.max()),
            "required_max_row_relative_l2": 1e-4,
        })
        if float(row_relative.max()) > 1e-4:
            raise AssertionError(json.dumps(results[-1], indent=2))
        if layer == 0:
            cached = {
                # Verification A uses eight rows, while Verification B requires
                # 64 real rows. Capture and persist the larger frozen prefix.
                "down_input": capture.down[layer][: max(rows, 64)].clone(),
                "q_from_folded_weights": q_from_weights.cpu(),
                "source": folded.source.cpu(),
                "signs": folded.signs.cpu(),
                "seed": torch.tensor(seed),
            }
        del factor, folded, signs, hidden, down, gate_new, up_new
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results, cached


class DownHooks:
    """Quantize the down_proj input under one of the three factorizations.

    ``old``       H S P G x with the sequential Householder product (the
                  pre-registered reference operator).
    ``online_q``  H G' (Q x) with Q applied online, so the only change from
                  ``folded`` is where Q x comes from.
    ``folded``    H G' x where x is already Q x because gate/up carry Q.
    """

    def __init__(self, model: torch.nn.Module, factors: dict[int, act.RotationFactor],
                 folded: dict[int, FoldedR4], mode: str, sample_rows: int = 64):
        self.model = model
        self.factors = factors
        self.folded = folded
        self.mode = mode
        self.sample_rows = sample_rows
        self.handles: list[Any] = []
        self.matched = 0
        self.total = 0

    def _hook(self, layer: int) -> Callable[..., tuple[torch.Tensor]]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[torch.Tensor]:
            value = inputs[0]
            fold = self.folded[layer]
            if self.mode == "old":
                transformed = self.factors[layer].apply(value.float(), fold.signs)
            elif self.mode == "online_q":
                transformed = fold.apply(fold.q_unfolded(value.float()))
            else:
                transformed = fold.apply(value.float())
            dequant, _, _, codes = base.dynamic_asym_int4(transformed, 128)
            if self.mode != "old" and layer == 0 and self.total < self.sample_rows * value.shape[-1]:
                rows = value.float().reshape(-1, value.shape[-1])[: self.sample_rows]
                if self.mode == "online_q":
                    original = rows
                else:
                    original = torch.empty_like(rows)
                    original[:, fold.source] = rows * fold.signs
                old = self.factors[layer].apply(original, fold.signs)
                _, _, _, old_codes = base.dynamic_asym_int4(old, 128)
                old_codes = old_codes.reshape(-1, value.shape[-1])
                new_codes = codes.reshape(-1, value.shape[-1])[: self.sample_rows]
                self.matched += int((new_codes == old_codes).sum())
                self.total += int(new_codes.numel())
            return (dequant.to(value.dtype),) + inputs[1:]
        return hook

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(self._hook(layer)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.no_grad()
def fold_down_weights(model: torch.nn.Module, factors: dict[int, act.RotationFactor],
                      signs: dict[int, torch.Tensor], row_batch: int) -> None:
    for layer, block in enumerate(model.model.layers):
        weight = block.mlp.down_proj.weight.detach()
        chunks = []
        for start in range(0, weight.shape[0], row_batch):
            chunks.append(factors[layer].apply(weight[start:start + row_batch].float(), signs[layer]).to(weight.dtype))
        weight.copy_(torch.cat(chunks, dim=0))


def verify_ppl(workdir: Path, seed: int, sequences: int, row_batch: int) -> dict[str, Any]:
    model_key = "llama32_3b"
    model_id = MODELS[model_key]["id"]
    tokens = base.prepare_token_chunks(model_id, "test", 0, sequences, 2048, workdir)
    model = base.load_model(model_id, workdir)
    device = next(model.parameters()).device
    factors: dict[int, act.RotationFactor] = {}
    folded: dict[int, FoldedR4] = {}
    signs: dict[int, torch.Tensor] = {}
    for layer in range(MODELS[model_key]["layers"]):
        factor = act.RotationFactor.load(factor_path(workdir, model_key, layer), device)
        sign = signs_for(factor.n, layer, seed, device)
        factors[layer] = factor
        signs[layer] = sign
        folded[layer] = FoldedR4.from_factor(factor, sign)
    fold_down_weights(model, factors, signs, row_batch)

    def evaluate(mode: str) -> tuple[float, float]:
        hooks = DownHooks(model, factors, folded, mode)
        hooks.install()
        try:
            losses = act.evaluate_nlls(model, tokens, f"E17v2 {mode} factorization")
        finally:
            hooks.close()
        match = hooks.matched / hooks.total if hooks.total else float("nan")
        return math.exp(sum(losses) / len(losses)), match

    old_ppl, _ = evaluate("old")
    online_ppl, online_match = evaluate("online_q")
    for layer, block in enumerate(model.model.layers):
        fold_swiglu_weights(block.mlp.gate_proj, block.mlp.up_proj, folded[layer])
    folded_ppl, folded_match = evaluate("folded")

    result = {
        "model": model_key,
        "site": "down_only",
        "rank": 8,
        "sequences": sequences,
        "sequence_length": 2048,
        "old_ppl": old_ppl,
        "online_q_ppl": online_ppl,
        "new_ppl": folded_ppl,
        # Pre-registered gate: folded factorization against the original
        # sequential-Householder operator.
        "ppl_abs_difference": abs(folded_ppl - old_ppl),
        "required_ppl_abs_difference": 1e-3,
        # The weight fold in isolation: identical operator, Q x produced by the
        # permuted gate/up weights instead of an online gather.
        "fold_only_ppl_abs_difference": abs(folded_ppl - online_ppl),
        # Re-factorization in isolation: sequential Householders to compact WY
        # with H distributed over the rank-k correction (the E12 change).
        "refactorization_ppl_abs_difference": abs(online_ppl - old_ppl),
        "sampled_code_match_fraction": folded_match,
        "online_q_sampled_code_match_fraction": online_match,
        "required_code_match_fraction": 0.999,
    }
    result["passed"] = (
        result["ppl_abs_difference"] <= result["required_ppl_abs_difference"]
        and result["sampled_code_match_fraction"] >= result["required_code_match_fraction"]
    )
    result["fold_only_passed"] = (
        result["fold_only_ppl_abs_difference"] <= result["required_ppl_abs_difference"]
        and result["sampled_code_match_fraction"] >= result["required_code_match_fraction"]
    )
    del model, factors, folded, signs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("fold verification requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e17-v2-fold-verification")
    output_path = workdir / "results" / "llama32_3b" / "e17v2_fold_verification.json"
    previous_ppl = None
    if output_path.exists():
        previous_ppl = json.loads(output_path.read_text()).get("ppl")
    fold_rows: list[dict[str, Any]] = []
    for model in args.models:
        rows, cached = verify_model(workdir, model, args.seed, args.rows)
        fold_rows.extend(rows)
        cache_path = workdir / "artifacts" / "e17v2" / model / "real_down_rows_layer00.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached, cache_path)
    result: dict[str, Any] = {
        "models": args.models,
        "rank": 8,
        "real_model_fold_rows": fold_rows,
        "ppl": previous_ppl,
        "hardware": base.hardware_info(),
    }
    if args.run_ppl:
        result["ppl"] = verify_ppl(workdir, args.seed, args.ppl_sequences, args.row_batch)
    base.atomic_json(output_path, result)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--rows", type=int, default=8)
    result.add_argument("--run-ppl", action="store_true")
    result.add_argument("--ppl-sequences", type=int, default=64)
    result.add_argument("--row-batch", type=int, default=256)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
