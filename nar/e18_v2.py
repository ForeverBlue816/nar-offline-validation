#!/usr/bin/env python3
"""E18 v2 - Qwen3 root-cause diagnosis and Base-model rerun.

The v1 Qwen3-8B result is not usable: NAR k=max beat the unquantized bf16 row
on 53 of 64 chunks.  This module runs the ordered diagnosis before any number
is reported.

  Step 0  per-chunk NLL in fp32 (the v1 path cast a bf16 loss after the fact).
  Step 1  rotation-only control with the identity quantizer; every rotated row
          must reproduce the bf16 row, otherwise the fold changed the network.
  Step 2  per-layer orthogonality and fold audits in fp32.
  Step 3a Qwen3 architecture audit (tying, q_norm/k_norm, biases, fold sites).
  Step 4  the Base-model rerun with the k-curve and the sqrt(1-f) prediction.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:
    from . import activation_experiments as act
    from . import e18_70b as e18
    from . import experiment as base
except ImportError:
    import activation_experiments as act
    import e18_70b as e18
    import experiment as base


LOG = logging.getLogger("nar")
GROUP = 128
MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
    # The E18 70B run reported a destroyed Hadamard row (PPL 15025 against a
    # bf16 3.105); it has never had the rotation-only control that localised
    # the Qwen3 defect, so the same control is run for it here.
    "llama31_70b": "unsloth/Meta-Llama-3.1-70B",
}
CONTROL_METHODS = ("hadamard", "nar_k8", "nar_kmax")
STEP4_RANKS: tuple[Any, ...] = (8, 16, 32, 64, "max")


def factor_root(workdir: Path, model_key: str, version: str) -> Path:
    return workdir / "activations" / model_key / f"{version}_factors"


# ---------------------------------------------------------------- Step 3a ---

def architecture_audit(model: torch.nn.Module, model_id: str) -> dict[str, Any]:
    """Audit every Qwen3 feature that a generic Llama fold path could break."""
    config = model.config
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = bool(getattr(config, "tie_word_embeddings", False))
    shared_storage = bool(
        head is not None and embed.weight.data_ptr() == head.weight.data_ptr()
    )
    biases: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            biases.append(name)
    norms = sorted({
        name.split(".")[-1] for name, module in model.named_modules()
        if module.__class__.__name__.endswith("RMSNorm")
    })
    per_head_norms = [
        name for name, module in model.named_modules()
        if module.__class__.__name__.endswith("RMSNorm") and name.endswith(("q_norm", "k_norm"))
    ]
    return {
        "model_id": model_id,
        "architecture": config.architectures[0] if config.architectures else None,
        "tie_word_embeddings": tied,
        "embedding_and_lm_head_share_storage": shared_storage,
        # E18 rotates activations at two sites and folds R into the *consuming*
        # linear's input axis.  No R1 residual-stream rotation, no embedding or
        # lm_head fold, and no R2 head_dim rotation exist in this protocol, so
        # weight tying cannot be double-applied here.
        "folds_embedding_or_lm_head": False,
        "residual_stream_rotation_r1": False,
        "head_dim_rotation_r2": False,
        "rmsnorm_gamma_fusion_performed": False,
        "rmsnorm_module_kinds": norms,
        "per_head_qk_norm_modules": len(per_head_norms),
        "per_head_qk_norm_left_untouched": True,
        "linear_modules_with_bias": biases,
        "attention_bias": bool(getattr(config, "attention_bias", False)),
        "rms_norm_eps": float(getattr(config, "rms_norm_eps", math.nan)),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "slot_counts": {
            "qkv": int(config.hidden_size) // GROUP,
            "down": int(config.intermediate_size) // GROUP,
        },
        "quantized_sites": ["input_layernorm output (q/k/v_proj input)", "down_proj input"],
        "folded_weights": ["q_proj", "k_proj", "v_proj", "down_proj"],
    }


@torch.no_grad()
def capture_reference(model: torch.nn.Module, tokens: torch.Tensor, rows: int
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 layer-0 output and final logits of the unmodified model."""
    device = e18._input_device(model)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        captured["layer0"] = (output[0] if isinstance(output, tuple) else output).detach().float()

    handle = model.model.layers[0].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            logits = model(input_ids=tokens[:1, :rows].to(device), use_cache=False).logits
        return captured["layer0"].clone(), logits.detach().float().clone()
    finally:
        handle.remove()


@torch.no_grad()
def fold_forward_check(model: torch.nn.Module, tokens: torch.Tensor,
                       reference: tuple[torch.Tensor, torch.Tensor], rows: int) -> dict[str, Any]:
    """Compare a rotated, unquantized model against the reference forward."""
    layer0_reference, logits_reference = reference
    device = e18._input_device(model)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        captured["layer0"] = (output[0] if isinstance(output, tuple) else output).detach().float()

    handle = model.model.layers[0].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            logits = model(input_ids=tokens[:1, :rows].to(device), use_cache=False).logits.detach().float()
    finally:
        handle.remove()
    layer0 = captured["layer0"]
    return {
        "check_rows": rows,
        "layer0_max_abs_difference": float((layer0 - layer0_reference).abs().max()),
        "layer0_relative_l2": float(
            (layer0 - layer0_reference).norm() / layer0_reference.norm().clamp_min(1e-30)
        ),
        "logits_max_abs_difference": float((logits - logits_reference).abs().max()),
        "logits_relative_l2": float(
            (logits - logits_reference).norm() / logits_reference.norm().clamp_min(1e-30)
        ),
        "required_relative_l2": 1e-3,
    }


# ----------------------------------------------------------------- Step 2 ---

def _spectral_norm(matrix: torch.Tensor, iterations: int = 64) -> float:
    """Power iteration for the spectral norm of a symmetric matrix."""
    vector = torch.randn(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    vector /= vector.norm()
    value = 0.0
    for _ in range(iterations):
        product = matrix @ vector
        value = float(product.norm())
        if value == 0.0:
            return 0.0
        vector = product / value
    return value


@torch.no_grad()
def orthogonality_rows(model_key: str, root: Path, method: str, site: str, n: int,
                       layers: list[int], seed: int, device: torch.device) -> list[dict[str, Any]]:
    """||R^T R - I|| and a condition-number bound for one method and site."""
    rows: list[dict[str, Any]] = []
    identity = torch.eye(n, device=device, dtype=torch.float32)
    for layer in layers:
        generator = torch.Generator(device="cpu").manual_seed(act._seed(seed, 0, layer, site))
        signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64)
        signs = signs.float().mul_(2).sub_(1).to(device)
        if method == "hadamard":
            rotated = act.full_hadamard_rows(identity, signs)
            rank = n // GROUP
            anchor = 0.0
        else:
            factor = e18.ShardedFactor(root / method / f"{site}_layer_{layer:02d}.pt", device)
            rotated = factor.apply(identity, signs)
            rank = int(factor.factor.active.sum())
            anchor = float(factor.factor.anchor_error)
            del factor
        # rows of `rotated` are R e_i, so gram = R^T R for the row convention.
        gram = rotated.T @ rotated
        deviation = gram - identity
        spectral = _spectral_norm(deviation)
        bound = math.sqrt((1.0 + spectral) / max(1.0 - spectral, 1e-30)) if spectral < 1 else math.inf
        rows.append({
            "model": model_key, "method": method, "site": site, "layer": layer,
            "n": n, "active_reflectors": rank, "anchor_error": anchor,
            "gram_max_abs_deviation": float(deviation.abs().max()),
            "gram_relative_frobenius": float(deviation.norm() / math.sqrt(n)),
            "gram_spectral_norm": spectral,
            "condition_number_bound": bound,
        })
        del rotated, gram, deviation
        gc.collect()
        torch.cuda.empty_cache()
    del identity
    return rows


# ------------------------------------------------------- exact fold path ---

def full_hadamard_rows_transpose(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Row-form transpose of act.full_hadamard_rows (same dispatch order).

    The forward operator is r -> ((r * s) H_q) H_p on the two factored axes, so
    the transpose applies the symmetric FWHT, then the Paley factor untransposed,
    and multiplies by the signs last.
    """
    n = x.shape[-1]
    quotient, remainder = divmod(n, 28)
    if not remainder and quotient >= 1 and not quotient & (quotient - 1):
        factored = act.ext._fast_walsh_hadamard(x.reshape(-1, 28, quotient))
        h28 = act.paley_hadamard_28(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h28).transpose(1, 2).reshape_as(x) * signs
    quotient, remainder = divmod(n, 12)
    if not remainder and quotient >= 1 and not quotient & (quotient - 1):
        factored = act.ext._fast_walsh_hadamard(x.reshape(-1, 12, quotient))
        h12 = act.ext._paley_hadamard_12(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h12).transpose(1, 2).reshape_as(x) * signs
    if not n & (n - 1):
        return act.ext._fast_walsh_hadamard(x) * signs
    if n == 3072:
        factored = act.ext._fast_walsh_hadamard(x.reshape(-1, 12, 256))
        h12 = act.ext._paley_hadamard_12(x.device, x.dtype)
        return (factored.transpose(1, 2) @ h12).transpose(1, 2).reshape_as(x) * signs
    raise ValueError(f"no transpose for full-Hadamard n={n}")


def factor_apply_transpose(sharded: "e18.ShardedFactor", value: torch.Tensor,
                           signs: torch.Tensor) -> torch.Tensor:
    """Transpose of ShardedFactor.apply: G^T P^-1 S H, with H symmetric."""
    factor = sharded.factor
    n = factor.n
    shape = value.shape
    rows = value.float().reshape(-1, n)
    rows = act.ext._fast_walsh_hadamard(rows.reshape(-1, n // GROUP, GROUP)).reshape(-1, n)
    rows = rows * signs
    unpermuted = torch.empty_like(rows)
    unpermuted[:, factor.source_order] = rows[:, factor.target_order]
    return (unpermuted - (unpermuted @ sharded.y) @ sharded.w.T).reshape(shape)


# ----------------------------------------------------------------- Step 1 ---

class ExactRotationHooks(e18.QuantHooks):
    """Fake-quantize as R^T Q(R x) and leave every weight untouched.

    Folding R into the consuming weight re-rounds R W to bf16, which perturbs
    the rotated rows by ~0.4% relative while the bf16 row carries no such
    perturbation.  Applying R^T on the activation is algebraically the same
    operator (W R^T R x = W x) but keeps all rows on identical weights and
    identical kernels, so the only difference measured is the quantizer.
    """

    def __init__(self, model: torch.nn.Module, rotations: "ShardedRotationsAt",
                 layers: int, quantize: bool = True):
        super().__init__(model, rotations, layers)
        self.quantize_values = quantize

    def quantize(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        rotated = self.rotations.apply(site, layer, value)
        if self.quantize_values:
            rotated, _, _, _ = base.dynamic_asym_int4(rotated, GROUP)
        return self.rotations.apply_transpose(site, layer, rotated).to(value.dtype)


class RotationHooks(e18.QuantHooks):
    """QuantHooks with the INT4 step optionally replaced by the identity."""

    def __init__(self, model: torch.nn.Module, rotations: "e18.ShardedRotations",
                 layers: int, quantize: bool = True):
        super().__init__(model, rotations, layers)
        self.quantize_values = quantize

    def quantize(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        rotated = self.rotations.apply(site, layer, value)
        if not self.quantize_values:
            return rotated.to(value.dtype)
        dequant, _, _, _ = base.dynamic_asym_int4(rotated, GROUP)
        return dequant.to(value.dtype)


def evaluate_method(model: torch.nn.Module, tokens: torch.Tensor, args: argparse.Namespace,
                    root: Path, method: str, dimensions: dict[str, int], layers: int,
                    weights: "e18.ShardedWeights | None", quantize: bool,
                    reference: tuple[torch.Tensor, torch.Tensor] | None = None,
                    ) -> tuple[list[float], float, dict[str, Any], "e18.ShardedWeights | None"]:
    """Evaluate one method row under the exact-fold or the v1 weight-fold path."""
    rotations = ShardedRotationsAt(args, model, root, method, dimensions, layers)
    error = 0.0
    if args.weight_fold:
        if weights is None:
            weights = e18.ShardedWeights(model, layers)
        weights.restore_all()
        error = weights.rotate_all(rotations, args.weight_row_batch)
        hooks: e18.QuantHooks = RotationHooks(model, rotations, layers, quantize=quantize)
    else:
        hooks = ExactRotationHooks(model, rotations, layers, quantize=quantize)
    extra: dict[str, Any] = {}
    if reference is not None:
        extra = {
            f"roundtrip_relative_error_{site}": rotations.roundtrip_error(
                site, 0, n, e18._device_of(model.model.layers[0].mlp.down_proj)
            )
            for site, n in dimensions.items()
        }
    hooks.install()
    label = f"{args.model_key} {method}" + ("" if quantize else " rotation-only")
    try:
        if reference is not None:
            extra.update(fold_forward_check(model, tokens, reference, args.check_rows))
        losses = e18.evaluate(model, tokens, label)
    finally:
        hooks.close()
        if args.weight_fold and weights is not None:
            weights.restore_all()
        del rotations, hooks
        gc.collect()
        torch.cuda.empty_cache()
    return losses, error, extra, weights


class ShardedRotationsAt(e18.ShardedRotations):
    """ShardedRotations that reads factors from an explicit root directory."""

    def __init__(self, args: argparse.Namespace, model: torch.nn.Module, root: Path,
                 method: str, dimensions: dict[str, int], layers: int):
        self.method = method
        self.signs: dict[tuple[str, int], torch.Tensor] = {}
        self.factors: dict[tuple[str, int], e18.ShardedFactor] = {}
        for layer, block in enumerate(model.model.layers[:layers]):
            for site, module in (("qkv", block.self_attn.q_proj), ("down", block.mlp.down_proj)):
                device = e18._device_of(module)
                generator = torch.Generator(device="cpu").manual_seed(act._seed(args.seed, 0, layer, site))
                signs = torch.randint(0, 2, (dimensions[site],), generator=generator, dtype=torch.int64)
                self.signs[(site, layer)] = signs.float().mul_(2).sub_(1).to(device)
                if method.startswith("nar_"):
                    self.factors[(site, layer)] = e18.ShardedFactor(
                        root / method / f"{site}_layer_{layer:02d}.pt", device
                    )

    def apply_transpose(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        signs = self.signs[(site, layer)]
        if self.method == "hadamard":
            return full_hadamard_rows_transpose(value.float(), signs)
        return factor_apply_transpose(self.factors[(site, layer)], value, signs)

    def roundtrip_error(self, site: str, layer: int, n: int, device: torch.device,
                        rows: int = 8) -> float:
        """||R^T R x - x|| / ||x||; guards the hand-derived transposes."""
        probe = torch.randn((rows, n), device=device, dtype=torch.float32)
        restored = self.apply_transpose(site, layer, self.apply(site, layer, probe))
        return float((restored - probe).norm() / probe.norm())


def paired_stats(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    deltas = [b - a for a, b in zip(reference, candidate)]
    mean = sum(deltas) / len(deltas)
    std = (sum((d - mean) ** 2 for d in deltas) / len(deltas)) ** 0.5
    return {
        "chunks": len(deltas),
        "chunks_below_bf16": sum(1 for d in deltas if d < 0),
        "mean_nll_delta": mean,
        "std_nll_delta": std,
        "max_abs_nll_delta": max(abs(d) for d in deltas),
    }


# ----------------------------------------------------------------- Step 4 ---

def f_of_k_rows(eigenspace_csv: Path, model_key: str, ranks: list[int]) -> list[dict[str, Any]]:
    """Mean captured energy f(k) per site with the 1 - sqrt(1-f) prediction."""
    raw = base.read_csv(eigenspace_csv)
    rows: list[dict[str, Any]] = []
    for site in sorted({str(row["site"]) for row in raw}):
        per_layer: dict[int, dict[int, float]] = {}
        for row in raw:
            if str(row["site"]) != site:
                continue
            per_layer.setdefault(int(row["layer"]), {})[int(row["rank"])] = float(
                row["cumulative_fraction_total_energy"]
            )
        available = max(max(values) for values in per_layer.values())
        for k in sorted({min(rank, available) for rank in ranks}):
            values = [layer_values[k] for layer_values in per_layer.values() if k in layer_values]
            if not values:
                continue
            mean = sum(values) / len(values)
            rows.append({
                "model": model_key, "site": site, "k": k, "layers": len(values),
                "mean_cumulative_captured_fraction": mean,
                "min_cumulative_captured_fraction": min(values),
                "max_cumulative_captured_fraction": max(values),
                "predicted_range_reduction": 1.0 - math.sqrt(max(0.0, 1.0 - mean)),
            })
    return rows


def full_split_sequences(model_id: str, split: str, seq_len: int, workdir: Path) -> int:
    """Number of non-overlapping context windows in one WikiText-2 split."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split,
        cache_dir=str(workdir / "cache" / "datasets"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=str(workdir / "cache" / "huggingface"), use_fast=True
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    total = len(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])
    return total // (seq_len - 1)


# -------------------------------------------------------------------- run ---

def run(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, f"e18v2-{args.model_key}")
    base.seed_everything(args.seed)
    model_id = MODELS[args.model_key]
    result_dir = workdir / "results" / args.model_key
    model = e18.load_sharded_model(model_id, workdir)
    layers = e18.selected_layers(model, args.max_layers)
    dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
    audit = architecture_audit(model, model_id)
    LOG.info("E18v2 architecture audit: %s", json.dumps(audit, indent=2))
    if audit["linear_modules_with_bias"]:
        raise AssertionError(f"unexpected biases in the fold path: {audit['linear_modules_with_bias']}")

    version = "e18v2" if args.recalibrate else "e18"
    root = factor_root(workdir, args.model_key, version)
    ranks = STEP4_RANKS if args.recalibrate else (8, "max")
    e18.calibrate(args, model, model_id, args.model_key, dimensions, layers, ranks=ranks,
                  root_override=root, eigenspace_name=f"{version}_calibration_eigenspace.csv")

    train_sequences = args.calibration_sequences
    if args.full_test_set:
        eval_sequences = full_split_sequences(model_id, "test", args.seq_len, workdir)
        LOG.info("full WikiText-2 test set yields %d context windows", eval_sequences)
    else:
        eval_sequences = args.eval_sequences
    tokens = base.prepare_token_chunks(model_id, "test", 0, eval_sequences, args.seq_len, workdir)
    # Calibration comes from the train split and evaluation from the test split,
    # so the two chunk sets cannot overlap; assert the splits really differ.
    if args.calibration_split == args.eval_split:
        raise AssertionError("calibration and evaluation must use disjoint splits")

    methods = ["bf16"] + [label for label, _ in e18.resolve_ranks(ranks, dimensions["down"] // GROUP)]
    if not args.skip_hadamard:
        methods.insert(1, "hadamard")

    weights: e18.ShardedWeights | None = None
    control_rows: list[dict[str, Any]] = []
    quantized: dict[str, list[float]] = {}
    fold_errors: dict[str, float] = {}

    bf16_losses = e18.evaluate(model, tokens, f"{args.model_key} bf16")
    quantized["bf16"] = bf16_losses
    bf16_ppl = math.exp(sum(bf16_losses) / len(bf16_losses))

    if args.rotation_only_control:
        reference = capture_reference(model, tokens, args.check_rows)
        for method in [m for m in CONTROL_METHODS if m in methods]:
            losses, error, extra, weights = evaluate_method(
                model, tokens, args, root, method, dimensions, layers, weights,
                quantize=False, reference=reference,
            )
            row = {
                "model": args.model_key, "model_id": model_id, "method": method,
                "quantizer": "identity",
                "fold": "weight_fold" if args.weight_fold else "exact_transpose",
                "seed": args.seed, "bf16_ppl": bf16_ppl,
                "rotation_only_ppl": math.exp(sum(losses) / len(losses)),
                "weight_fold_max_relative_error": error,
                "required_ppl_abs_difference": 0.01,
                "required_max_abs_nll_delta": 1e-3,
                **extra, **paired_stats(bf16_losses, losses),
            }
            row["ppl_abs_difference"] = abs(row["rotation_only_ppl"] - row["bf16_ppl"])
            row["passed"] = bool(row["ppl_abs_difference"] <= 0.01
                                 and row["max_abs_nll_delta"] <= 1e-3)
            control_rows.append(row)
        base.write_csv(result_dir / "e18v2_rotation_only_control.csv", control_rows)
        LOG.info("Step 1 control: %s", json.dumps(control_rows, indent=2))
        failed = [row["method"] for row in control_rows if not row["passed"]]
        if failed and args.report_only:
            # Diagnostic mode: record the arm and continue, so a later arm of
            # the same investigation still runs. Never used for a reported row.
            LOG.warning("rotation-only control did not pass for %s; continuing (report-only)", failed)
        elif args.require_control and failed:
            raise AssertionError("Step 1 rotation-only control failed; refusing to report E18 v2")

    if args.orthogonality_audit:
        audit_layers = sorted({0, layers // 2, layers - 1})
        audit_rows: list[dict[str, Any]] = []
        for method in [m for m in CONTROL_METHODS if m in methods]:
            for site, n in dimensions.items():
                audit_rows.extend(orthogonality_rows(
                    args.model_key, root, method, site, n, audit_layers, args.seed,
                    torch.device("cuda"),
                ))
        base.write_csv(result_dir / "e18v2_orthogonality_audit.csv", audit_rows)
        audit["orthogonality_audit_layers"] = audit_layers
        audit["max_gram_deviation"] = max(row["gram_max_abs_deviation"] for row in audit_rows)

    controls: dict[str, list[float]] = {}
    if args.evaluate:
        for method in methods:
            if method == "bf16":
                continue
            # Paired identity-quantizer control for the SAME rotation.  The
            # rotation round trip leaves a small rank-dependent offset, so the
            # quantization cost is only clean against this control, not bf16.
            controls[method], _, _, weights = evaluate_method(
                model, tokens, args, root, method, dimensions, layers, weights, quantize=False,
            )
            losses, error, _, weights = evaluate_method(
                model, tokens, args, root, method, dimensions, layers, weights, quantize=True,
            )
            quantized[method] = losses
            fold_errors[method] = error
        rows = [
            {"model": args.model_key, "model_id": model_id, "site": "both", "method": method,
             "seed": args.seed, "sequence": index, "nll": value,
             "tokens_scored": args.seq_len - 1,
             "rotation_only_nll": controls.get(method, values)[index] if method in controls else value,
             "weight_fold_max_relative_error": fold_errors.get(method, 0.0)}
            for method, values in quantized.items() for index, value in enumerate(values)
        ]
        if any(not isinstance(row["nll"], float) for row in rows):
            raise AssertionError("per-chunk NLL must be written as fp32 python floats")
        base.write_csv(result_dir / "e18v2_per_sequence.csv", rows)
        def mean_ppl(values: list[float]) -> float:
            return math.exp(sum(values) / len(values))

        had_ppl = mean_ppl(quantized["hadamard"]) if "hadamard" in quantized else math.nan
        # Control-corrected quantization cost: PPL of the row minus the PPL of
        # the same rotation with the identity quantizer.
        had_cost = (had_ppl - mean_ppl(controls["hadamard"])) if "hadamard" in controls else math.nan
        summary = []
        for method, values in quantized.items():
            ppl = mean_ppl(values)
            control_ppl = mean_ppl(controls[method]) if method in controls else bf16_ppl
            cost = ppl - control_ppl
            summary.append({
                "model": args.model_key, "site": "both", "method": method, "seed": args.seed,
                "ppl": ppl, "rotation_only_ppl": control_ppl,
                "ppl_delta_vs_bf16": ppl - bf16_ppl,
                "rotation_only_delta_vs_bf16": control_ppl - bf16_ppl,
                "quantization_cost": cost,
                "ppl_delta_vs_hadamard": math.nan if method == "bf16" else ppl - had_ppl,
                "recovered_fraction_of_hadamard_gap": (
                    math.nan if method in ("bf16", "hadamard") or not math.isfinite(had_cost) or had_cost == 0
                    else (had_cost - cost) / had_cost
                ),
                "effective_bits_per_value": 16.0 if method == "bf16" else 4.25,
                **paired_stats(quantized["bf16"], values),
                **{f"vs_control_{key}": value
                   for key, value in paired_stats(controls.get(method, values), values).items()},
            })
        base.write_csv(result_dir / "e18v2_summary.csv", summary)
        base.write_csv(
            result_dir / "e18v2_f_of_k.csv",
            f_of_k_rows(result_dir / f"{version}_calibration_eigenspace.csv", args.model_key,
                        [rank for _, rank in e18.resolve_ranks(ranks, dimensions["down"] // GROUP)]),
        )

    base.atomic_json(result_dir / "e18v2_fold_audit.json", {
        "architecture_audit": audit,
        "rotation_only_control": control_rows,
        "factor_root": str(root),
        "hardware": base.hardware_info(),
    })
    if args.evaluate:
        base.atomic_json(result_dir / "E18V2_DONE.json", {
            "model": args.model_key, "model_id": model_id, "seed": args.seed,
            "num_layers": layers, "ranks": list(ranks),
            "eval_split": args.eval_split, "eval_sequences": eval_sequences,
            "full_test_set": bool(args.full_test_set), "sequence_length": args.seq_len,
            "calibration_split": args.calibration_split,
            "calibration_sequences": train_sequences,
            "nll_dtype": "float32",
            "fold": "weight_fold" if args.weight_fold else "exact_transpose",
            "architecture_audit": audit,
            "rotation_only_control": control_rows,
            "weight_fold_max_relative_error": fold_errors,
            "activation_quantizer": "dynamic asymmetric per-token group-128 INT4",
            "sites": "post-RMSNorm q/k/v_proj input and down_proj input only",
            "hardware": base.hardware_info(),
        })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--model-key", choices=tuple(MODELS), default="qwen3_8b")
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--calibration-sequences", type=int, default=128)
    result.add_argument("--eval-sequences", type=int, default=64)
    result.add_argument("--seq-len", type=int, default=2048)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--oversample", type=int, default=16)
    result.add_argument("--permutation-stride", type=int, default=32)
    result.add_argument("--weight-row-batch", type=int, default=256)
    result.add_argument("--max-layers", type=int, default=0)
    result.add_argument("--calibration-split", default="train")
    result.add_argument("--eval-split", default="test")
    result.add_argument("--rotation-only-control", action="store_true")
    result.add_argument("--orthogonality-audit", action="store_true")
    result.add_argument("--evaluate", action="store_true")
    result.add_argument("--recalibrate", action="store_true",
                        help="build the full k-curve factor set under e18v2_factors")
    result.add_argument("--full-test-set", action="store_true")
    result.add_argument("--skip-hadamard", action="store_true")
    result.add_argument("--weight-fold", action="store_true",
                        help="reproduce the v1 path that folds R into the consuming weight")
    result.add_argument("--report-only", action="store_true",
                        help="record a failing control instead of aborting; diagnostics only")
    result.add_argument("--require-control", action="store_true",
                        help="abort unless every rotation-only row reproduces bf16")
    result.add_argument("--check-rows", type=int, default=64,
                        help="token positions used for the fp32 layer-0 and logits fold checks")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
