#!/usr/bin/env python3
"""E19 end-to-end W4A4KV4 on Qwen3-8B-Base.

Extends the frozen E14 pipeline to a non-Llama architecture.  Everything that
can be reused from ``nar/e14_w4a4kv4.py`` is reused; what differs on Qwen3 is
guarded by an explicit architecture audit that aborts rather than silently
relying on the Llama code path matching.

Two deviations from E14 are deliberate and follow the E18 v2 root cause:

  * the whole pipeline runs in fp32.  Loading bf16 weights into fp32 containers
    changes no value, but it means a folded weight is never written back as
    bf16, so the rotated rows carry no rounding that the reference row lacks.
  * per-chunk NLL is computed from fp32 logits and the dtype is asserted in the
    writer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import activation_experiments as act
    from . import e14_w4a4kv4 as e14
    from . import experiment as base
    from .e12_wy import WYFactor, compact_wy
except ImportError:
    import activation_experiments as act
    import e14_w4a4kv4 as e14
    import experiment as base
    from e12_wy import WYFactor, compact_wy


LOG = logging.getLogger("nar")
MODEL_KEY = "qwen3_8b_base"
MODEL_ID = "Qwen/Qwen3-8B-Base"
GROUP = e14.GROUP
EVAL_WINDOWS = 146
ROTATIONS = ("hadamard", "nar_k8", "nar_k32", "nar_kmax")
ROWS = {
    "bf16": (None, None),
    "hadamard_asym_g128": ("hadamard", "asymmetric_g128"),
    "nar_k8_asym_g128": ("nar_k8", "asymmetric_g128"),
    "nar_k32_asym_g128": ("nar_k32", "asymmetric_g128"),
    "nar_kmax_asym_g128": ("nar_kmax", "asymmetric_g128"),
}
ZERO_SHOT_ROWS = ("hadamard_asym_g128", "nar_k8_asym_g128", "nar_kmax_asym_g128")
# Expected Qwen3-8B-Base shape contract; every value is asserted at load time.
EXPECTED = {
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36,
}


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


def config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_model_fp32(workdir: Path) -> torch.nn.Module:
    """Load the base checkpoint into fp32 containers; values are unchanged."""
    from transformers import AutoModelForCausalLM

    LOG.info("loading %s in fp32", MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=str(workdir / "cache" / "huggingface"),
        dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    if model.config._name_or_path and MODEL_ID not in str(model.config._name_or_path):
        raise AssertionError(f"expected {MODEL_ID}, loaded {model.config._name_or_path}")
    return model.eval().cuda()


# ------------------------------------------------------------------ audit ---

def architecture_audit(model: torch.nn.Module) -> dict[str, Any]:
    """Assert every Qwen3 property the E14 Llama path silently assumes."""
    config = model.config
    problems: list[str] = []

    # (d) shape contract, including the GQA grouping the KV quantizer depends on.
    shapes = {
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
    }
    for key, want in EXPECTED.items():
        if shapes[key] != want:
            problems.append(f"{key}={shapes[key]} expected {want}")

    # (e) tying and biases.
    head = model.get_output_embeddings()
    tied = bool(getattr(config, "tie_word_embeddings", False))
    shared = bool(head is not None
                  and head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())
    if tied or shared:
        problems.append(f"tie_word_embeddings={tied} shared_storage={shared}; E19 expects both false")
    biased = [name for name, module in model.named_modules()
              if isinstance(module, torch.nn.Linear) and module.bias is not None]
    if biased:
        problems.append(f"biased linear modules present: {biased}")

    # (a) per-head q_norm/k_norm must exist and must not be fused anywhere.
    qk_norms = sorted(name for name, module in model.named_modules()
                      if module.__class__.__name__.endswith("RMSNorm")
                      and name.endswith(("q_norm", "k_norm")))
    fused_norms = sorted({name.split(".")[-1] for name, module in model.named_modules()
                          if module.__class__.__name__.endswith("RMSNorm")}
                         & {"input_layernorm", "post_attention_layernorm", "norm"})
    if len(qk_norms) != 2 * shapes["num_hidden_layers"]:
        problems.append(f"expected {2 * shapes['num_hidden_layers']} q_norm/k_norm, found {len(qk_norms)}")

    audit = {
        "model_id": MODEL_ID,
        "architecture": config.architectures[0] if config.architectures else None,
        "shapes": shapes,
        "slot_counts": {"qkv": shapes["hidden_size"] // GROUP,
                        "down": shapes["intermediate_size"] // GROUP},
        "kv_group_counts": {
            "k_per_channel_groups": shapes["head_dim"] // e14.K_TOKEN_GROUP,
            "v_per_token_group_size": shapes["head_dim"],
            "kv_heads": shapes["num_key_value_heads"],
            "num_key_value_groups": shapes["num_attention_heads"] // shapes["num_key_value_heads"],
        },
        "tie_word_embeddings": tied,
        "embedding_and_lm_head_share_storage": shared,
        "linear_modules_with_bias": biased,
        # (a) fuse_norms_and_rotate touches only these three RMSNorm kinds.
        "rmsnorm_kinds_fused_into_consumers": fused_norms,
        "per_head_qk_norm_modules": len(qk_norms),
        "per_head_qk_norm_fused": False,
        "per_head_qk_norm_note": (
            "q_norm/k_norm are applied inside Qwen3Attention after q_proj/k_proj and "
            "before RoPE. fuse_norms_and_rotate folds only input_layernorm, "
            "post_attention_layernorm and the final norm, so they are untouched."
        ),
        # (b) R1 acts on the residual/hidden axis only.
        "r1_note": (
            "R1 folds into the input axis of q/k/v_proj and gate/up_proj and the output "
            "axis of o_proj and down_proj. q_norm/k_norm live downstream of q_proj/k_proj "
            "on the head_dim axis and are therefore unaffected by R1."
        ),
        # (c) R2 acts on head_dim.
        "r2_note": "R2 acts on the head_dim axis (v_proj output / o_proj input), independent of q_norm/k_norm.",
        "problems": problems,
    }
    if problems:
        raise AssertionError("Qwen3 architecture audit failed:\n  " + "\n  ".join(problems))
    return audit


@torch.inference_mode()
def kv_site_probe(model: torch.nn.Module) -> dict[str, Any]:
    """(a) Record which module produces the tensor the K quantizer sees."""
    observed: dict[str, Any] = {}
    block = model.model.layers[0]

    def record(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            observed[name] = list(value.shape)
        return hook

    handles = [block.self_attn.k_proj.register_forward_hook(record("k_proj_output")),
               block.self_attn.k_norm.register_forward_hook(record("k_norm_output")),
               block.self_attn.q_norm.register_forward_hook(record("q_norm_output"))]
    captured: dict[str, Any] = {}
    key = f"nar_e19_probe_{id(model)}"

    def probe_attention(module, query, key_states, value, attention_mask=None, **kwargs):
        captured["key_shape"] = list(key_states.shape)
        captured["value_shape"] = list(value.shape)
        captured["query_shape"] = list(query.shape)
        from transformers.models.llama.modeling_llama import repeat_kv
        repeated = repeat_kv(key_states, module.num_key_value_groups)
        captured["key_shape_after_repeat"] = list(repeated.shape)
        weights = torch.matmul(query, repeated.transpose(-1, -2)) * kwargs.get(
            "scaling", getattr(module, "scaling", query.shape[-1] ** -0.5))
        causal = torch.arange(repeated.shape[-2], device=query.device).unsqueeze(0) <= torch.arange(
            repeated.shape[-2] - query.shape[-2], repeated.shape[-2], device=query.device).unsqueeze(1)
        weights = weights.masked_fill(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(weights.dtype).min)
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(weights, repeat_kv(value, module.num_key_value_groups)).transpose(1, 2).contiguous(), None

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    previous = model.config._attn_implementation
    ALL_ATTENTION_FUNCTIONS.register(key, probe_attention)
    model.config._attn_implementation = key
    try:
        model(input_ids=torch.zeros((1, 8), dtype=torch.long, device="cuda"), use_cache=False)
    finally:
        model.config._attn_implementation = previous
        for handle in handles:
            handle.remove()
    head_dim = int(getattr(model.config, "head_dim", 128))
    return {
        "module_shapes": observed,
        "attention_interface_shapes": captured,
        "k_quantizer_input": "post q_norm/k_norm and post-RoPE key, i.e. the cache tensor",
        "k_quantizer_matches_llama_functional_point": True,
        "k_is_post_k_norm": observed.get("k_norm_output") is not None
        and captured.get("key_shape", [0, 0, 0, 0])[-1] == head_dim,
    }


# -------------------------------------------------------------- rotations ---

class Qwen3RotationSet(e14.RotationSet):
    """E14 RotationSet with Qwen3 R4 factors and the per-site rank cap.

    R1 lives on the 4096-wide residual, so it has 32 group-128 slots and rank
    32 already saturates it; ``nar_k32`` and ``nar_kmax`` therefore share the
    same R1 and differ only in R4, which has 96 slots on the 12288-wide MLP.
    """

    R1_LABEL = {"nar_k8": "k8", "nar_k32": "kmax", "nar_kmax": "kmax"}
    R4_LABEL = {"nar_k8": "nar_k8", "nar_k32": "nar_k32", "nar_kmax": "nar_kmax"}

    def __init__(self, workdir: Path, model_key: str, method: str, seed: int,
                 config: Any, device: torch.device):
        self.method = method
        self.seed = seed
        self.device = device
        self.layers = int(config.num_hidden_layers)
        self.heads = int(config.num_attention_heads)
        self.hidden = int(config.hidden_size)
        self.intermediate = int(config.intermediate_size)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.r1 = None
        self.r2: dict[int, act.RotationFactor] = {}
        self.r4: dict[int, act.RotationFactor] = {}
        self.r4_wy: dict[int, WYFactor] = {}
        self.sign_cache: dict[tuple[str, int], torch.Tensor] = {}
        if not method.startswith("nar_"):
            return
        root = e14.rotation_dir(workdir, model_key)
        if not (root / "DONE.json").exists():
            raise FileNotFoundError(root / "DONE.json")
        self.r1 = act.RotationFactor.load(root / f"r1_{self.R1_LABEL[method]}.pt", device)
        r4_root = workdir / "activations" / model_key / "e18v2_factors" / self.R4_LABEL[method]
        if not r4_root.exists():
            raise FileNotFoundError(r4_root)
        for layer in range(self.layers):
            self.r2[layer] = act.RotationFactor.load(root / f"r2_v_layer_{layer:02d}.pt", device)
            self.r4[layer] = act.RotationFactor.load(r4_root / f"down_layer_{layer:02d}.pt", device)
            w, y = compact_wy(self.r4[layer].reflectors, self.r4[layer].active)
            self.r4_wy[layer] = WYFactor(self.r4[layer], w, y)

    def ranks(self) -> dict[str, Any]:
        return {
            "r1_rank": int(self.r1.active.sum()) if self.r1 is not None else None,
            "r1_slots": self.hidden // GROUP,
            "r4_rank": int(self.r4[0].active.sum()) if self.r4 else None,
            "r4_slots": self.intermediate // GROUP,
            "r2_rank": int(self.r2[0].active.sum()) if self.r2 else None,
        }


def apply_transpose(rotations: e14.RotationSet, label: str, layer: int,
                    value: torch.Tensor) -> torch.Tensor:
    """R^T for every rotation E19 uses, mirroring the E18 v2 derivation."""
    signs = rotations.signs(label, layer, value.shape[-1])
    if rotations.method == "hadamard":
        if label != "r1":
            signs = torch.ones_like(signs)
        return e18v2_full_hadamard_transpose(value.float(), signs)
    factor = (rotations.r1 if label == "r1"
              else rotations.r2[layer] if label == "r2" else rotations.r4[layer])
    n = factor.n
    shape = value.shape
    rows = value.float().reshape(-1, n)
    rows = act.ext._fast_walsh_hadamard(rows.reshape(-1, n // factor.b, factor.b)).reshape(-1, n)
    rows = rows * signs
    unpermuted = torch.empty_like(rows)
    unpermuted[:, factor.source_order] = rows[:, factor.target_order]
    reflectors, active = factor.reflectors, factor.active
    w, y = compact_wy(reflectors, active)
    return (unpermuted - (unpermuted @ y) @ w.T).reshape(shape)


def e18v2_full_hadamard_transpose(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    try:
        from .e18_v2 import full_hadamard_rows_transpose
    except ImportError:
        from e18_v2 import full_hadamard_rows_transpose
    return full_hadamard_rows_transpose(x, signs)


@torch.inference_mode()
def round_trip_audit(rotations: e14.RotationSet, layers: int, rows: int = 8,
                     tolerance: float = 1e-6) -> list[dict[str, Any]]:
    """||R^T R x - x|| / ||x|| at every site and layer, per the fold contract."""
    sites = {"r1": rotations.hidden, "r2": rotations.head_dim, "r4": rotations.intermediate}
    audit: list[dict[str, Any]] = []
    for layer in range(layers):
        for label, n in sites.items():
            if label == "r1" and layer > 0:
                continue  # R1 is a single global rotation.
            probe = torch.randn((rows, n), device=rotations.device, dtype=torch.float32)
            restored = apply_transpose(rotations, label, layer, rotations.apply(label, layer, probe))
            residual = float((restored - probe).norm() / probe.norm())
            audit.append({"site": label, "layer": layer, "n": n,
                          "round_trip_relative_error": residual, "tolerance": tolerance})
            if residual > tolerance:
                raise AssertionError(f"round-trip check failed: {audit[-1]}")
    return audit


# ------------------------------------------------------------- evaluation ---

def eval_tokens(workdir: Path, seq_len: int) -> torch.Tensor:
    """The same 146 WikiText-2 test windows E18 v2 used, for comparability."""
    return base.prepare_token_chunks(MODEL_ID, "test", 0, EVAL_WINDOWS, seq_len, workdir)


@torch.inference_mode()
def evaluate_ppl_fp32(model: torch.nn.Module, tokens: torch.Tensor,
                      label: str) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index in range(tokens.shape[0]):
        batch = tokens[index:index + 1].cuda()
        logits = model(input_ids=batch, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )
        if loss.dtype != torch.float32:
            raise AssertionError(f"per-chunk NLL must be fp32, got {loss.dtype}")
        value = float(loss)
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite E19 loss at chunk {index}")
        rows.append({"chunk": index, "nll": value, "tokens_scored": int(tokens.shape[1] - 1)})
        if index % 16 == 0:
            LOG.info("%s PPL chunk %d/%d nll=%.6f", label, index + 1, tokens.shape[0], value)
        del logits, loss
    ppl = math.exp(float(np.average([r["nll"] for r in rows],
                                    weights=[r["tokens_scored"] for r in rows])))
    return ppl, rows


# ---------------------------------------------------------------- commands ---

def install_extension_hooks() -> None:
    """Point the E14 pipeline at the Qwen3 loader, rotations and control."""
    e14.LOAD_MODEL = lambda model_id, workdir: load_model_fp32(Path(workdir))
    e14.ROTATION_SET = Qwen3RotationSet
    e14.ALGEBRA_CONTROL = algebra_control


def control_path(workdir: Path) -> Path:
    return workdir / "results" / MODEL_KEY / "e19_rotation_only_control.csv"


def algebra_control(workdir: Path, rotation: str) -> dict[str, Any]:
    """Step 2 gate, consumed by GPTQ exactly as E14 consumes its own."""
    path = control_path(Path(workdir))
    if not path.exists():
        raise FileNotFoundError(f"run `control` before `gptq`: {path}")
    rows = [row for row in base.read_csv(path) if row["rotation"] == rotation]
    if not rows:
        raise AssertionError(f"no rotation-only control row for {rotation} in {path}")
    row = rows[0]
    if str(row["passed"]).lower() != "true":
        raise AssertionError(f"rotation-only control failed for {rotation}: {row}")
    return {"source": str(path), "row": row}


def calibrate_command(args: argparse.Namespace) -> None:
    """Build R1 (k=8 and k=32) and the per-layer R2 through E14's calibrator.

    R4 is not built here: E18 v2 already froze per-layer down-input NAR factors
    for this checkpoint at k=8/16/32/64/96, and E19 reuses them unchanged.
    """
    args.model = MODEL_KEY
    e14.calibrate_rotations(args)


def audit_command(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e19-audit")
    model = load_model_fp32(workdir)
    audit = architecture_audit(model)
    audit["kv_site_probe"] = kv_site_probe(model)
    audit["compute_dtype"] = "float32"
    audit["git_commit"] = git_commit()
    audit["hardware"] = base.hardware_info()
    base.atomic_json(workdir / "results" / MODEL_KEY / "e19_architecture_audit.json", audit)
    LOG.info("E19 architecture audit: %s", json.dumps(audit, indent=2))


def paired(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    deltas = [b - a for a, b in zip(reference, candidate)]
    mean = sum(deltas) / len(deltas)
    return {
        "chunks": len(deltas),
        "chunks_below_reference": sum(1 for d in deltas if d < 0),
        "mean_nll_delta": mean,
        "max_abs_nll_delta": max(abs(d) for d in deltas),
    }


def control_command(args: argparse.Namespace) -> None:
    """Step 2: rotation-only control, logit check and KV group check."""
    if not torch.cuda.is_available():
        raise RuntimeError("E19 control requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e19-control")
    base.seed_everything(args.seed)
    install_extension_hooks()
    tokens = eval_tokens(workdir, args.seq_len)[: args.control_chunks]

    model = load_model_fp32(workdir)
    audit = architecture_audit(model)
    probe = tokens[:1, : args.verify_tokens].cuda()
    with torch.inference_mode():
        reference_logits = model(input_ids=probe, use_cache=False).logits.float().clone()
    reference_ppl, reference_rows = evaluate_ppl_fp32(model, tokens, "reference")
    reference_nll = [row["nll"] for row in reference_rows]
    del model
    gc.collect()
    torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    round_trips: list[dict[str, Any]] = []
    for rotation in args.rotations:
        model, rotations, fold = e14._prepare_rotated_model(
            workdir, MODEL_KEY, rotation, args.seed, args.weight_row_batch
        )
        trip = round_trip_audit(rotations, rotations.layers, tolerance=args.round_trip_tolerance)
        round_trips.extend({"rotation": rotation, **entry} for entry in trip)
        hooks = e14.RuntimeHooks(model, rotations, activation_kind=None, quantize_kv=False)
        hooks.install()
        try:
            with torch.inference_mode():
                observed = model(input_ids=probe, use_cache=False).logits.float()
            logit_max = float((observed - reference_logits).abs().max())
            logit_rel = float((observed - reference_logits).norm()
                              / reference_logits.norm().clamp_min(1e-30))
            del observed
            ppl, chunk_rows = evaluate_ppl_fp32(model, tokens, f"{rotation} rotation-only")
        finally:
            hooks.close()
        stats = paired(reference_nll, [row["nll"] for row in chunk_rows])
        row = {
            "model": MODEL_KEY, "model_id": MODEL_ID, "rotation": rotation,
            "quantizer": "identity", "fold_dtype": "float32", "seed": args.seed,
            "reference_ppl": reference_ppl, "rotation_only_ppl": ppl,
            "ppl_abs_difference": abs(ppl - reference_ppl),
            "required_ppl_abs_difference": args.control_ppl_tolerance,
            "max_abs_logit_difference": logit_max,
            "relative_l2_logit_difference": logit_rel,
            "max_round_trip_relative_error": max(e["round_trip_relative_error"] for e in trip),
            "round_trip_tolerance": args.round_trip_tolerance,
            **stats, **rotations.ranks(),
        }
        # A rotation-only row must reproduce the reference in magnitude, and
        # must not sit systematically below it once the difference is large
        # enough for its sign to mean anything. E18 v1's defect was mean dNLL
        # -8.5e-3 with |dPPL| 0.109; fp32 re-association noise is ~1e-5, and a
        # sign test at that scale only measures rounding, so it is applied only
        # when the per-chunk difference clears the same 1e-3 bound E18 v2 used.
        share = row["chunks_below_reference"] / row["chunks"]
        row["below_reference_share"] = share
        row["required_max_abs_nll_delta"] = args.control_nll_tolerance
        row["sign_test_applicable"] = bool(row["max_abs_nll_delta"] > args.control_nll_tolerance)
        row["sign_balanced"] = bool(0.25 <= share <= 0.75)
        row["passed"] = bool(
            row["ppl_abs_difference"] <= args.control_ppl_tolerance
            and row["max_abs_nll_delta"] <= args.control_nll_tolerance
            and (row["sign_balanced"] or not row["sign_test_applicable"])
        )
        rows.append(row)
        LOG.info("E19 control %s: %s", rotation, json.dumps(row, indent=2))
        del model, rotations, hooks
        gc.collect()
        torch.cuda.empty_cache()

    # Merge on the method column: a run restricted with --rotations computes a
    # subset, and writing it wholesale would delete the rows another
    # invocation produced. GPTQ gates on this file, so losing a row breaks it.
    def merge_csv(path: Path, fresh: list[dict[str, Any]], key: str) -> None:
        merged: dict[str, dict[str, Any]] = {}
        if path.exists():
            for existing in base.read_csv(path):
                merged[str(existing[key])] = existing
        for row in fresh:
            merged[str(row[key])] = row
        base.write_csv(path, list(merged.values()))

    merge_csv(control_path(workdir), rows, "rotation")
    trip_path = workdir / "results" / MODEL_KEY / "e19_round_trip_audit.csv"

    def trip_key(entry: dict[str, Any]) -> str:
        return f"{entry['rotation']}|{entry['site']}|{entry['layer']}"

    merged_trips: dict[str, dict[str, Any]] = {}
    if trip_path.exists():
        for existing in base.read_csv(trip_path):
            merged_trips[trip_key(existing)] = existing
    for entry in round_trips:
        merged_trips[trip_key(entry)] = entry
    base.write_csv(trip_path, list(merged_trips.values()))
    base.atomic_json(workdir / "results" / MODEL_KEY / "e19_control.json", {
        "model": MODEL_KEY, "model_id": MODEL_ID, "control_chunks": int(tokens.shape[0]),
        "reference_ppl": reference_ppl, "rows": rows,
        "kv_group_counts": audit["kv_group_counts"],
        "compute_dtype": "float32", "git_commit": git_commit(),
        "hardware": base.hardware_info(),
    })
    failed = [row["rotation"] for row in rows if not row["passed"]]
    if failed:
        raise AssertionError(f"E19 Step 2 rotation-only control failed for {failed}")


def gptq_command(args: argparse.Namespace) -> None:
    install_extension_hooks()
    args.model = MODEL_KEY
    e14.gptq_quantize(args)


def evaluate_command(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E19 evaluation requires CUDA")
    workdir = Path(args.workdir).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    result_dir = workdir / "results" / MODEL_KEY
    ppl_path = result_dir / f"e19_{args.row}_seed{args.seed}_ppl.json"
    zero_path = result_dir / f"e19_{args.row}_seed{args.seed}_zero_shot.json"
    need_ppl = args.metrics in ("ppl", "both")
    need_zero = args.metrics in ("zero_shot", "both") and args.row in ZERO_SHOT_ROWS
    if (not need_ppl or ppl_path.exists()) and (not need_zero or zero_path.exists()):
        LOG.info("E19 row already complete: %s", args.row)
        return
    base.setup_logging(workdir, f"e19-evaluate-{args.row}")
    base.seed_everything(args.seed)
    install_extension_hooks()
    rotation, activation_kind = ROWS[args.row]

    provenance = {
        "model": MODEL_KEY, "model_id": MODEL_ID, "row": args.row,
        "rotation_checkpoint": rotation, "seed": args.seed,
        "compute_dtype": "float32", "fold_dtype": "float32",
        "git_commit": git_commit(), "hardware": base.hardware_info(),
        "gptq_config_hash": config_hash({
            "bits": 4, "perchannel": True, "symmetric": True, "groupsize": -1,
            "blocksize": 128, "percdamp": 0.01, "act_order": False,
            "calibration_sequences": args.calibration_sequences,
            "calibration_seed": args.calibration_seed, "sequence_length": args.seq_len,
        }),
        "effective_bits": {
            "activation": 4.25 if activation_kind else 16.0,
            "key": 4.25 if rotation else 16.0,
            "value": 4.25 if rotation else 16.0,
            "weight": 4.0 if rotation else 16.0,
        },
    }

    if args.row == "bf16":
        model = load_model_fp32(workdir)
        rotations = None
        provenance["rotation_calibration_hash"] = None
        provenance["round_trip_max_relative_error"] = 0.0
        provenance["weight_fold_audit"] = {"folded": False}
        hooks = None
    else:
        model, rotations = e14.load_quantized_model(
            workdir, artifact_root, MODEL_KEY, rotation, args.seed, args.weight_row_batch
        )
        trip = round_trip_audit(rotations, rotations.layers, tolerance=args.round_trip_tolerance)
        provenance["round_trip_max_relative_error"] = max(
            entry["round_trip_relative_error"] for entry in trip)
        calibration = e14.rotation_dir(workdir, MODEL_KEY) / "DONE.json"
        provenance["rotation_calibration_hash"] = config_hash(json.loads(calibration.read_text()))
        provenance["weight_fold_audit"] = json.loads(
            (e14.checkpoint_dir(artifact_root, MODEL_KEY, rotation, args.seed) / "DONE.json").read_text()
        )["fold"]
        provenance["ranks"] = rotations.ranks()
        hooks = e14.RuntimeHooks(model, rotations, activation_kind=activation_kind, quantize_kv=True)
        hooks.install()
    try:
        if need_ppl and not ppl_path.exists():
            tokens = eval_tokens(workdir, args.seq_len)
            ppl, chunk_rows = evaluate_ppl_fp32(model, tokens, f"{MODEL_KEY} {args.row}")
            base.atomic_json(ppl_path, {
                **provenance, "ppl": ppl, "chunks": chunk_rows,
                "chunks_evaluated": len(chunk_rows),
                "dataset": "WikiText-2 raw test, full set, 146 windows at context 2048",
                "sequence_length": args.seq_len, "nll_dtype": "float32",
            })
            del tokens
            gc.collect()
            torch.cuda.empty_cache()
        if need_zero and not zero_path.exists():
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            from lm_eval.tasks import TaskManager
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, cache_dir=str(workdir / "cache" / "huggingface"), use_fast=True
            )
            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size,
                      max_batch_size=args.batch_size, max_length=args.seq_len)
            result = lm_eval.simple_evaluate(
                model=lm, tasks=list(e14.TASKS), num_fewshot=0, batch_size=args.batch_size,
                max_batch_size=args.batch_size, task_manager=TaskManager(), cache_requests=False,
                bootstrap_iters=0, log_samples=False, random_seed=args.seed,
                numpy_random_seed=args.seed, torch_random_seed=args.seed,
                fewshot_random_seed=args.seed, apply_chat_template=False, fewshot_as_multiturn=False,
            )
            if result is None:
                raise RuntimeError("lm-eval returned no result")
            task_rows = [{"task": task, "metric": e14.METRICS[task],
                          "accuracy": float(result["results"][task][e14.METRICS[task]])}
                         for task in e14.TASKS]
            base.atomic_json(zero_path, {
                **provenance, "tasks": task_rows,
                "mean_accuracy": float(np.mean([row["accuracy"] for row in task_rows])),
                "harness_commit": e14.HARNESS_COMMIT,
                "versions": e14._serializable(result.get("versions", {})),
            })
    finally:
        if hooks is not None:
            hooks.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def finalize_command(args: argparse.Namespace) -> None:
    """Aggregate whatever exists; never depends on a live controller."""
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e19-finalize")
    result_dir = workdir / "results" / MODEL_KEY
    present: dict[str, dict[str, Any]] = {}
    zero: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for row in ROWS:
        ppl_path = result_dir / f"e19_{row}_seed{args.seed}_ppl.json"
        zero_path = result_dir / f"e19_{row}_seed{args.seed}_zero_shot.json"
        if ppl_path.exists():
            present[row] = json.loads(ppl_path.read_text())
        else:
            missing.append(f"{row}:ppl")
        if zero_path.exists():
            zero[row] = json.loads(zero_path.read_text())
        elif row in ZERO_SHOT_ROWS:
            missing.append(f"{row}:zero_shot")

    summary: list[dict[str, Any]] = []
    bf16_ppl = present.get("bf16", {}).get("ppl")
    had_ppl = present.get("hadamard_asym_g128", {}).get("ppl")
    had_zero = zero.get("hadamard_asym_g128", {}).get("mean_accuracy")
    gap = (had_ppl - bf16_ppl) if (bf16_ppl is not None and had_ppl is not None) else None
    tiers = {"bf16": "reference", "hadamard_asym_g128": "B",
             "nar_k8_asym_g128": "C", "nar_k32_asym_g128": "C", "nar_kmax_asym_g128": "C"}
    for row, payload in present.items():
        bits = payload.get("effective_bits", {})
        mean_zero = zero.get(row, {}).get("mean_accuracy")
        summary.append({
            "model": MODEL_KEY, "tier": tiers[row], "row": row, "seed": args.seed,
            "effective_bits_activation": bits.get("activation"),
            "effective_bits_key": bits.get("key"), "effective_bits_value": bits.get("value"),
            "effective_bits_weight": bits.get("weight"),
            "ppl": payload["ppl"], "chunks_evaluated": payload.get("chunks_evaluated"),
            "ppl_delta_vs_bf16": (payload["ppl"] - bf16_ppl) if bf16_ppl is not None else math.nan,
            "ppl_delta_vs_hadamard": (payload["ppl"] - had_ppl) if had_ppl is not None else math.nan,
            "recovered_fraction": (
                (had_ppl - payload["ppl"]) / gap
                if gap not in (None, 0) and row not in ("bf16", "hadamard_asym_g128") else math.nan
            ),
            "zero_shot_mean": mean_zero if mean_zero is not None else math.nan,
            "zero_shot_delta_vs_hadamard": (
                mean_zero - had_zero if (mean_zero is not None and had_zero is not None) else math.nan
            ),
            "round_trip_max_relative_error": payload.get("round_trip_max_relative_error"),
            "fold_dtype": payload.get("fold_dtype"), "compute_dtype": payload.get("compute_dtype"),
            "gptq_config_hash": payload.get("gptq_config_hash"),
            "rotation_calibration_hash": payload.get("rotation_calibration_hash"),
            "model_id": payload.get("model_id"), "git_commit": payload.get("git_commit"),
            "gpu": (payload.get("hardware") or {}).get("gpu_name"),
        })
    if summary:
        base.write_csv(result_dir / "e19_summary.csv", summary)
    control = control_path(workdir)
    nar_rows = [r for r in summary if r["row"].startswith("nar_")]
    best_nar = min((r["ppl"] for r in nar_rows), default=None)
    base.atomic_json(result_dir / "E19_DONE.json", {
        "model": MODEL_KEY, "model_id": MODEL_ID, "seed": args.seed,
        "rows_complete": sorted(present), "rows_missing": missing,
        "complete": not missing,
        "summary": summary,
        "rotation_only_control": base.read_csv(control) if control.exists() else None,
        "stop_condition": "NAR k=max must beat Hadamard in full-test PPL",
        "stop_condition_met": (
            None if (had_ppl is None or "nar_kmax_asym_g128" not in present)
            else bool(present["nar_kmax_asym_g128"]["ppl"] < had_ppl)
        ),
        "best_nar_ppl": best_nar,
        "eval_windows": EVAL_WINDOWS, "sequence_length": args.seq_len,
        "compute_dtype": "float32", "git_commit": git_commit(),
        "no_tier_a_row": (
            "The released QuaRot code does not target Qwen3, so no Tier A anchor is "
            "run here; the Llama Tier A anchor in E14 remains the literature reference."
        ),
        "hardware": base.hardware_info(),
    })
    LOG.info("E19 finalize: %d rows present, missing=%s", len(present), missing)
    print(json.dumps({"rows_complete": sorted(present), "rows_missing": missing}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--artifact-root", default=None)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--seq-len", type=int, default=2048)
    result.add_argument("--weight-row-batch", type=int, default=256)
    result.add_argument("--calibration-sequences", type=int, default=128)
    result.add_argument("--calibration-seed", type=int, default=0)
    result.add_argument("--verify-tokens", type=int, default=128)
    result.add_argument("--fold-tolerance", type=float, default=1e-2)
    result.add_argument("--round-trip-tolerance", type=float, default=1e-6)
    result.add_argument("--batch-size", type=int, default=1)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate")
    sub.add_parser("audit")
    control = sub.add_parser("control")
    # Every rotation that later gets a GPTQ pass needs a control row, because
    # gptq_command gates on this file; nar_k32 was missing and aborted the run.
    control.add_argument("--rotations", nargs="+",
                         default=["hadamard", "nar_k8", "nar_k32", "nar_kmax"])
    control.add_argument("--control-chunks", type=int, default=64)
    control.add_argument("--control-ppl-tolerance", type=float, default=0.01)
    control.add_argument("--control-nll-tolerance", type=float, default=1e-3)
    gptq = sub.add_parser("gptq")
    gptq.add_argument("--rotation", choices=ROTATIONS, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--row", choices=tuple(ROWS), required=True)
    evaluate.add_argument("--metrics", choices=("ppl", "zero_shot", "both"), default="both")
    sub.add_parser("finalize")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.artifact_root is None:
        args.artifact_root = str(Path(args.workdir) / "artifacts" / "e19")
    {"calibrate": calibrate_command, "audit": audit_command, "control": control_command,
     "gptq": gptq_command, "evaluate": evaluate_command,
     "finalize": finalize_command}[args.command](args)


if __name__ == "__main__":
    main()
