"""E21 - end-to-end W4A4KV4 on Llama-3.1-70B.

E14 established the end-to-end pipeline on Llama-3.1-8B and E19 carried it to a
second architecture family.  Neither could run at 70B, because E14 loads the
whole model onto one GPU and 70B does not fit.  This module supplies the three
things the 70B needs and reuses E14 for everything else:

  * a loader that shards the model across the visible GPUs for calibration and
    evaluation, and one that leaves it on the CPU for GPTQ, whose layer loop
    already moves a single decoder layer to the GPU at a time;
  * a rotation set that replicates its factors per device, since a sharded
    model presents activations on several of them;
  * the R4 root, because E14 reuses frozen E5/E11 per-layer down-input factors
    and the 70B has no E11 run.  E18's 70B down factors carry exactly the
    fields RotationFactor.load reads, so they are used directly rather than
    recalibrated.

Precision.  The whole pipeline runs in fp32 containers holding the published
bf16 values.  This is not a preference: the rotation-only control at 70B sits at
2.4e-3 on the per-chunk gate of 1e-3 in bf16 and passes at 1e-6 to 7e-5 in fp32,
a three-order-of-magnitude difference, so bf16 cannot certify the algebra here.

Rows.  bf16 reference, NAR k=8 and NAR k=max.  There is deliberately no Hadamard
row, so no paired delta against a metadata-matched baseline is available and
every claim here is a degradation against the model's own bf16 reference.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nar import activation_experiments as act  # noqa: E402
from nar import e14_w4a4kv4 as e14  # noqa: E402
from nar import experiment as base  # noqa: E402

LOG = base.LOG

MODEL_KEY = "llama31_70b"
MODEL_ID = "unsloth/Meta-Llama-3.1-70B"
EVAL_WINDOWS = 0          # 0 means the full contiguous WikiText-2 test stream
GROUP = e14.GROUP

ROWS: dict[str, tuple[str | None, str | None]] = {
    "bf16": (None, None),
    "nar_k8_asym_g128": ("nar_k8", "asymmetric_g128"),
    "nar_kmax_asym_g128": ("nar_kmax", "asymmetric_g128"),
}
ROTATIONS = ("nar_k8", "nar_kmax")
EXPECTED = {
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128,
    "hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ loaders ---

def _max_memory() -> dict[int, str]:
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("E21 requires CUDA")
    totals = [torch.cuda.get_device_properties(i).total_memory // 2**30 for i in range(count)]
    # Room for activations, logits, and the fold's temporary buffers.
    return {i: f"{max(8, gib - 12)}GiB" for i, gib in enumerate(totals)}


def load_model_sharded(model_id: str, workdir: Path) -> torch.nn.Module:
    """fp32 containers, sharded across every visible GPU."""
    from transformers import AutoModelForCausalLM

    memory = _max_memory()
    LOG.info("E21 loading %s in fp32 across %d GPUs: %s", model_id, len(memory), memory)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=str(workdir / "cache" / "huggingface"),
        dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="sdpa",
        device_map="balanced", max_memory=memory,
    )
    return model.eval()


def load_model_cpu(model_id: str, workdir: Path) -> torch.nn.Module:
    """fp32 containers on the CPU, for the GPTQ layer loop."""
    from transformers import AutoModelForCausalLM

    LOG.info("E21 loading %s in fp32 on the CPU for GPTQ", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=str(workdir / "cache" / "huggingface"),
        dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    return model.eval()


# ---------------------------------------------------------------- rotations ---

def _factor_to(factor: act.RotationFactor, device: torch.device) -> act.RotationFactor:
    return act.RotationFactor(
        n=factor.n, b=factor.b,
        reflectors=factor.reflectors.to(device), active=factor.active.to(device),
        source_order=factor.source_order.to(device), target_order=factor.target_order.to(device),
        anchor_error=factor.anchor_error,
    )


class ShardedRotationSet(e14.RotationSet):
    """E14's rotation set with per-device copies of every factor.

    A sharded model fires each hook on the device that owns the module, so a
    single copy on cuda:0 would either fail or force every activation across the
    interconnect.  Copies are made once per device and cached; on a single-GPU
    model exactly one copy exists and this is the base class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._r1: dict[torch.device, act.RotationFactor] = {}
        self._r2: dict[tuple[torch.device, int], act.RotationFactor] = {}
        self._r4: dict[tuple[torch.device, int], e14.WYFactor] = {}
        self._signs: dict[tuple[str, int, torch.device], torch.Tensor] = {}

    def ranks(self) -> dict[str, Any]:
        return {
            "r1_rank": int(self.r1.active.sum()) if self.r1 is not None else None,
            "r1_slots": self.hidden // GROUP,
            "r4_rank": int(self.r4[0].active.sum()) if self.r4 else None,
            "r4_slots": self.intermediate // GROUP,
            "r2_rank": int(self.r2[0].active.sum()) if self.r2 else None,
        }

    def signs(self, label: str, layer: int, n: int) -> torch.Tensor:
        """Kept for callers that hold a device already; apply() routes per device."""
        return self._signs_on(label, layer, n, self.device)

    def _signs_on(self, label: str, layer: int, n: int, device: torch.device) -> torch.Tensor:
        key = (label, layer, device)
        if key not in self._signs:
            self._signs[key] = e14._signs(n, e14._seed(self.seed, label, layer), device)
        return self._signs[key]

    def apply(self, label: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        device = value.device
        signs = self._signs_on(label, layer, value.shape[-1], device)
        if self.method == "hadamard":
            if label != "r1":
                signs = torch.ones_like(signs)
            return act.full_hadamard_rows(value.float(), signs)
        if label == "r1":
            assert self.r1 is not None
            if device not in self._r1:
                self._r1[device] = _factor_to(self.r1, device)
            return self._r1[device].apply(value, signs)
        if label == "r2":
            key = (device, layer)
            if key not in self._r2:
                self._r2[key] = _factor_to(self.r2[layer], device)
            return self._r2[key].apply(value, signs)
        key = (device, layer)
        if key not in self._r4:
            factor = _factor_to(self.r4[layer], device)
            w, y = e14.compact_wy(factor.reflectors, factor.active)
            self._r4[key] = e14.WYFactor(factor, w, y)
        return self._r4[key].apply(value, signs)


def r4_root(workdir: Path, model_key: str, rank_label: str) -> Path:
    """E18's 70B per-layer down-input factors, which E14's R4 can read as-is."""
    method = f"nar_{rank_label}"
    root = workdir / "activations" / model_key / "e18_factors" / method
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def install_extension_hooks(for_gptq: bool = False) -> None:
    e14.LOAD_MODEL = load_model_cpu if for_gptq else load_model_sharded
    e14.ROTATION_SET = ShardedRotationSet
    e14.R4_ROOT = r4_root
    # The fold streams each weight chunk to the rotation's device; on the CPU
    # path that is the one GPU GPTQ uses, on the sharded path it is cuda:0.
    e14.FOLD_DEVICE = torch.device("cuda:0")


# -------------------------------------------------------------------- audit ---

def architecture_audit(model: torch.nn.Module) -> dict[str, Any]:
    config = model.config
    problems: list[str] = []
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
    head = model.get_output_embeddings()
    tied = bool(getattr(config, "tie_word_embeddings", False))
    shared = bool(head is not None
                  and head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())
    if tied or shared:
        problems.append(f"tie_word_embeddings={tied} shared_storage={shared}")
    biased = [name for name, module in model.named_modules()
              if isinstance(module, torch.nn.Linear) and module.bias is not None]
    if biased:
        problems.append(f"biased linear modules: {biased}")
    devices = sorted({str(p.device) for p in model.parameters()})
    return {
        "model": MODEL_KEY, "model_id": MODEL_ID, "architecture": type(model).__name__,
        "shapes": shapes, "tie_word_embeddings": tied,
        "embedding_and_lm_head_share_storage": shared,
        "linear_modules_with_bias": biased,
        "parameter_devices": devices, "shards": len(devices),
        "slot_counts": {"qkv": shapes["hidden_size"] // GROUP,
                        "down": shapes["intermediate_size"] // GROUP},
        "compute_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "problems": problems, "git_commit": git_commit(), "hardware": base.hardware_info(),
    }


# ------------------------------------------------------------------ tokens ---

def eval_tokens(workdir: Path, seq_len: int) -> torch.Tensor:
    return e14._full_wikitext_tokens(MODEL_ID, workdir, seq_len)


@torch.inference_mode()
def evaluate_ppl_fp32(model: torch.nn.Module, tokens: torch.Tensor,
                      label: str) -> tuple[float, list[dict[str, Any]]]:
    """Per-chunk NLL in fp32, matching E19 rather than E14's bf16 loss."""
    rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    for index in range(tokens.shape[0]):
        batch = tokens[index:index + 1].to(device)
        logits = model(input_ids=batch, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1).to(logits.device),
        )
        value = float(loss)
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite E21 loss at chunk {index}")
        rows.append({"chunk": index, "nll": value, "tokens_scored": tokens.shape[1] - 1})
        if index % 16 == 0:
            LOG.info("%s PPL chunk %d/%d nll=%.6f", label, index + 1, tokens.shape[0], value)
    ppl = math.exp(float(np.average([r["nll"] for r in rows],
                                    weights=[r["tokens_scored"] for r in rows])))
    return ppl, rows


# ---------------------------------------------------------------- commands ---

def audit_command(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e21-audit")
    install_extension_hooks()
    model = load_model_sharded(MODEL_ID, workdir)
    audit = architecture_audit(model)
    LOG.info("E21 architecture audit: %s", json.dumps(audit, indent=2))
    base.atomic_json(workdir / "results" / MODEL_KEY / "e21_architecture_audit.json", audit)
    if audit["problems"]:
        raise AssertionError(f"E21 architecture audit failed: {audit['problems']}")


def calibrate_command(args: argparse.Namespace) -> None:
    """R1 and R2 via E14's own calibration; R4 comes from E18's 70B factors."""
    install_extension_hooks()
    args.model = MODEL_KEY
    e14.calibrate_rotations(args)


def gptq_command(args: argparse.Namespace) -> None:
    install_extension_hooks(for_gptq=True)
    args.model = MODEL_KEY
    e14.gptq_quantize(args)


def effective_bits(activation_kind: str | None, quantize_weights: bool,
                   quantize_kv: bool, context: int = 2048) -> dict[str, float]:
    """Derived from the constants the run uses, never hard-coded."""
    residual = e14.KV_RESIDUAL_LENGTH
    quantized = max(0, context - residual)
    head_dim = EXPECTED["head_dim"]
    hidden, inter = EXPECTED["hidden_size"], EXPECTED["intermediate_size"]
    kv_dim = EXPECTED["num_key_value_heads"] * head_dim
    weight_values = 2 * hidden * hidden + 2 * kv_dim * hidden + 3 * inter * hidden
    weight_scales = 3 * hidden + 2 * kv_dim + 2 * inter
    return {
        "activation": (4 + 32 / GROUP) if activation_kind else 16.0,
        "weight": (4 + 16 * weight_scales / weight_values) if quantize_weights else 16.0,
        "key": ((quantized * (4 + 32 / e14.K_TOKEN_GROUP) + residual * 16) / context
                if quantize_kv else 16.0),
        "value": ((quantized * (4 + 32 / head_dim) + residual * 16) / context
                  if quantize_kv else 16.0),
        "context": context,
    }


def evaluate_command(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E21 evaluation requires CUDA")
    workdir = Path(args.workdir).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    result_dir = workdir / "results" / MODEL_KEY
    ppl_path = result_dir / f"e21_{args.row}_seed{args.seed}_ppl.json"
    if ppl_path.exists():
        LOG.info("E21 row already complete: %s", args.row)
        return
    base.setup_logging(workdir, f"e21-evaluate-{args.row}")
    base.seed_everything(args.seed)
    install_extension_hooks()
    rotation, activation_kind = ROWS[args.row]

    provenance = {
        "model": MODEL_KEY, "model_id": MODEL_ID, "row": args.row,
        "rotation_checkpoint": rotation, "seed": args.seed,
        "compute_dtype": "float32", "fold_dtype": "float32",
        "git_commit": git_commit(), "hardware": base.hardware_info(),
        "effective_bits": effective_bits(activation_kind, rotation is not None,
                                         rotation is not None, args.seq_len),
        "baseline_note": ("no Hadamard row was run for E21, so every number here is a "
                          "degradation against this model's own bf16 reference and no "
                          "paired delta against a metadata-matched rotation exists"),
    }

    hooks = None
    if rotation is None:
        model = load_model_sharded(MODEL_ID, workdir)
    else:
        model, rotations = e14.load_quantized_model(
            workdir, artifact_root, MODEL_KEY, rotation, args.seed, args.weight_row_batch
        )
        done = json.loads((e14.checkpoint_dir(artifact_root, MODEL_KEY, rotation, args.seed)
                           / "DONE.json").read_text())
        provenance["weight_fold_audit"] = done["fold"]
        provenance["gptq"] = done["gptq"]
        provenance["ranks"] = rotations.ranks()
        hooks = e14.RuntimeHooks(model, rotations, activation_kind=activation_kind,
                                 quantize_kv=True)
        hooks.install()
    try:
        tokens = eval_tokens(workdir, args.seq_len)
        ppl, chunk_rows = evaluate_ppl_fp32(model, tokens, f"{MODEL_KEY} {args.row}")
        base.atomic_json(ppl_path, {
            **provenance, "ppl": ppl, "chunks": chunk_rows,
            "chunks_evaluated": len(chunk_rows), "nll_dtype": "float32",
            "dataset": "WikiText-2 raw test full contiguous token stream",
            "sequence_length": args.seq_len,
        })
        LOG.info("E21 %s ppl=%.6f over %d windows", args.row, ppl, len(chunk_rows))
    finally:
        if hooks is not None:
            hooks.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def finalize_command(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e21-finalize")
    result_dir = workdir / "results" / MODEL_KEY
    present: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for row in ROWS:
        path = result_dir / f"e21_{row}_seed{args.seed}_ppl.json"
        if path.exists():
            present[row] = json.loads(path.read_text())
        else:
            missing.append(row)
    bf16 = present.get("bf16", {}).get("ppl")
    summary = []
    for row, payload in present.items():
        rotation, kind = ROWS[row]
        bits = effective_bits(kind, rotation is not None, rotation is not None, args.seq_len)
        summary.append({
            "model": MODEL_KEY, "row": row, "seed": args.seed,
            "effective_bits_activation": bits["activation"],
            "effective_bits_weight": bits["weight"],
            "effective_bits_key": bits["key"], "effective_bits_value": bits["value"],
            "ppl": payload["ppl"], "chunks_evaluated": payload.get("chunks_evaluated"),
            "ppl_delta_vs_bf16": (payload["ppl"] - bf16) if bf16 is not None else math.nan,
            "relative_degradation": ((payload["ppl"] - bf16) / bf16
                                     if bf16 else math.nan),
            "compute_dtype": payload.get("compute_dtype"),
            "model_id": payload.get("model_id"), "git_commit": payload.get("git_commit"),
        })
    if summary:
        base.write_csv(result_dir / "e21_summary.csv", summary)
    base.atomic_json(result_dir / "E21_DONE.json", {
        "model": MODEL_KEY, "model_id": MODEL_ID,
        "rows_complete": sorted(present), "rows_missing": sorted(missing),
        "hadamard_row": "not run; no metadata-matched paired delta is available",
        "compute": "fp32 containers holding the published bf16 values, sharded across GPUs",
        "r4_factors": "E18 per-layer down-input factors, reused rather than recalibrated",
        "git_commit": git_commit(), "hardware": base.hardware_info(),
    })
    print(json.dumps({"rows_complete": sorted(present), "rows_missing": sorted(missing)}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--artifact-root", default=None)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--seq-len", type=int, default=2048)
    result.add_argument("--calibration-sequences", type=int, default=128)
    result.add_argument("--calibration-seed", type=int, default=0)
    result.add_argument("--weight-row-batch", type=int, default=256)
    result.add_argument("--verify-tokens", type=int, default=64)
    result.add_argument("--fold-tolerance", type=float, default=0.02)
    result.add_argument("--batch-size", type=int, default=1)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("calibrate")
    gptq = sub.add_parser("gptq")
    gptq.add_argument("--rotation", choices=ROTATIONS, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--row", choices=tuple(ROWS), required=True)
    sub.add_parser("finalize")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.artifact_root is None:
        args.artifact_root = str(Path(args.workdir) / "artifacts" / "e21")
    {"audit": audit_command, "calibrate": calibrate_command, "gptq": gptq_command,
     "evaluate": evaluate_command, "finalize": finalize_command}[args.command](args)


if __name__ == "__main__":
    main()
