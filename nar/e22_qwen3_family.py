"""E22 — the Qwen3 Base family under W4A4KV4 with g128_asym weights.

Models: Qwen3-0.6B/1.7B/4B/8B/14B Base. Rows per model: bf16, Hadamard,
PrismQuant k=8, PrismQuant k=max, all under the protocol E14/E19 fixed —
asymmetric g128 activations at the seven sites, the E14 KIVI cache policy,
fp32 fold and fp32 containers, exact-transpose control — with GPTQ weights
under ``g128_asym`` (QuaRot's asymmetric per-group branch, 4.15625 bits).

Everything model-specific is delegated to E19, which already carries the
Qwen3 audit, rotation set, control, fold and fp32 evaluation for the 8B; this
module points E19 at a different checkpoint (``configure``), adds the R4
calibration E19 borrowed from E18 v2, the benchmark suite the Qwen3 technical
report evaluates base models on, and the generation gate.

Qwen3-32B-Base is not a public checkpoint (the Hub returns 404), so the
family stops at 14B; GPQA is a gated dataset and runs only once a token with
access is present.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nar import activation_experiments as act  # noqa: E402
from nar import e14_w4a4kv4 as e14  # noqa: E402
from nar import e18_70b as e18  # noqa: E402
from nar import e19_qwen3_e2e as e19  # noqa: E402
from nar import experiment as base  # noqa: E402

LOG = logging.getLogger("nar")
PREFIX = "e22"
PROTOCOL = "g128_asym"
ACTIVATION_KIND = "asymmetric_g128"
FAMILY = ("qwen3_0.6b_base", "qwen3_1.7b_base", "qwen3_4b_base", "qwen3_8b_base", "qwen3_14b_base")
ROWS = {
    "bf16": None,
    "hadamard_asym_g128": "hadamard",
    "nar_k8_asym_g128": "nar_k8",
    "nar_kmax_asym_g128": "nar_kmax",
}
ROTATIONS = ("hadamard", "nar_k8", "nar_kmax")
C4_WINDOWS = 256
C4_SHARD = "en/c4-validation.00000-of-00008.json.gz"

# Qwen3 technical report (arXiv 2505.09388), Tables 4-8, Base models. The
# report evaluates MMLU 5-shot, MMLU-Redux 5-shot, BBH 3-shot CoT, GSM8K
# 4-shot CoT, MATH 4-shot CoT and GPQA 5-shot CoT; its text does not say
# whether MATH is the full test set or MATH-500, nor which GPQA split.
TECH_REPORT = {
    "qwen3_0.6b_base": {"mmlu": 52.81, "mmlu_redux": 51.26, "bbh": 41.47, "gpqa": 26.77, "gsm8k": 59.59, "math": 32.44},
    "qwen3_1.7b_base": {"mmlu": 62.63, "mmlu_redux": 61.66, "bbh": 54.47, "gpqa": 28.28, "gsm8k": 75.44, "math": 43.50},
    "qwen3_4b_base": {"mmlu": 72.99, "mmlu_redux": 72.79, "bbh": 72.59, "gpqa": 36.87, "gsm8k": 87.79, "math": 54.10},
    "qwen3_8b_base": {"mmlu": 76.89, "mmlu_redux": 76.17, "bbh": 78.40, "gpqa": 44.44, "gsm8k": 89.84, "math": 60.80},
    "qwen3_14b_base": {"mmlu": 81.05, "mmlu_redux": 79.88, "bbh": 81.07, "gpqa": 39.90, "gsm8k": 92.49, "math": 62.02},
    "qwen3_32b_base": {"mmlu": 83.61, "mmlu_redux": 83.41, "bbh": 87.38, "gpqa": 49.49, "gsm8k": 93.40, "math": 61.62},
}
ANCHOR_FLAG_POINTS = 2.0

# Harness benchmarks. num_fewshot None means the task carries fixed in-prompt
# exemplars (BBH 3-shot CoT, MATH 4-shot CoT). gsm8k_cot ships eight fixed
# CoT exemplars with a first_n sampler, so num_fewshot=4 takes the first four.
# gen_kwargs override the task's generation settings and are recorded.
BENCHMARKS: dict[str, dict[str, Any]] = {
    "wikitext": {"kind": "ppl"},
    "c4": {"kind": "ppl"},
    "mmlu": {"kind": "harness", "tasks": ["mmlu"], "num_fewshot": 5, "headline": ("mmlu", "acc,none")},
    "mmlu_redux": {"kind": "harness", "tasks": ["mmlu_redux_generative"], "num_fewshot": 5,
                   "gen_kwargs": {"max_gen_toks": 8, "until": ["</s>", "\n"]},
                   "headline": ("mmlu_redux_generative", "exact_match,default")},
    "bbh": {"kind": "harness", "tasks": ["bbh_cot_fewshot"], "num_fewshot": None,
            "headline": ("bbh_cot_fewshot", "exact_match,get-answer")},
    # This harness revision's default max_gen_toks is 2048; a sequence that
    # never emits a stop string then runs to 2048 tokens and, in a batch,
    # holds the whole batch there. CoT answers on these tasks are a few
    # hundred tokens, so each generative benchmark carries an explicit cap.
    "gsm8k": {"kind": "harness", "tasks": ["gsm8k_cot"], "num_fewshot": 4,
              "gen_kwargs": {"max_gen_toks": 512},
              "headline": ("gsm8k_cot", "exact_match,flexible-extract")},
    "math": {"kind": "harness", "tasks": ["minerva_math"], "num_fewshot": None,
             "gen_kwargs": {"max_gen_toks": 512},
             "headline": ("minerva_math", "exact_match,none")},
    "math500": {"kind": "harness", "tasks": ["minerva_math500"], "num_fewshot": None,
                "gen_kwargs": {"max_gen_toks": 512},
                "headline": ("minerva_math500", "exact_match,none")},
    "gpqa": {"kind": "harness", "tasks": ["gpqa_main_cot_n_shot"], "num_fewshot": 5,
             "gen_kwargs": {"max_gen_toks": 768},
             "headline": ("gpqa_main_cot_n_shot", "exact_match,strict-match")},
    "eight_task": {"kind": "harness", "tasks": list(e14.EIGHT_TASKS), "num_fewshot": 0,
                   "headline": ("__mean__", "eight-task mean")},
    # The supplementary zero-shot task that replaces BBH across the family:
    # one of the eight Llama tasks, the largest of them (10,042 items, so a
    # standard error near 0.45 points) and the commonsense complement to the
    # reasoning-heavy rest of the suite.
    "hellaswag": {"kind": "harness", "tasks": ["hellaswag"], "num_fewshot": 0,
                  "headline": ("hellaswag", "acc_norm,none")},
}
# BBH is dropped and MATH is MATH-500 at every size (decision of 2026-09-07):
# the full MATH test set and BBH were 11,500 of the 13,300 long generations
# per row. "math" resolves to minerva_math500 for every family member.
DEFAULT_BENCHMARKS = ("wikitext", "c4", "mmlu", "gsm8k", "mmlu_redux", "gpqa", "hellaswag", "math")
MATH_BENCHMARK = {key: "math500" for key in FAMILY} | {"qwen3_32b_base": "math500"}
DEFAULT_BATCH = {"qwen3_0.6b_base": 16, "qwen3_1.7b_base": 16, "qwen3_4b_base": 12,
                 "qwen3_8b_base": 8, "qwen3_14b_base": 4}


# --------------------------------------------------------------- plumbing ---

def configure(model_key: str) -> None:
    """Point E19 (and through it E14) at one Qwen3 checkpoint."""
    if model_key not in FAMILY:
        raise ValueError(f"unknown E22 model {model_key}")
    model_id, _ = act.model_id_and_key(model_key)
    e19.MODEL_KEY = model_key
    e19.MODEL_ID = model_id
    e19.PREFIX = PREFIX
    e19.EXPECTED = _expected_shapes(model_id)
    e19.install_extension_hooks()


def _expected_shapes(model_id: str) -> dict[str, int]:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, cache_dir=str(WORKDIR / "cache" / "huggingface"))
    shapes = {
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
    }
    if shapes["head_dim"] != 128 or shapes["hidden_size"] % 128 or shapes["intermediate_size"] % 128:
        raise AssertionError(f"E22 assumes head_dim 128 and group-128-divisible widths: {shapes}")
    return shapes


WORKDIR = Path(".")


def result_dir(model_key: str) -> Path:
    path = WORKDIR / "results" / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_root() -> Path:
    return WORKDIR / "artifacts" / "e22"


def batch_size_for(args: argparse.Namespace) -> int | str:
    """An explicit batch size, or the harness's own probe capped per model.

    Log-likelihood batches carry full-vocabulary fp32 logits, so sixteen of
    the longest 5-shot MMLU prompts need 28 GB on top of the model; "auto"
    probes the longest request first and picks what fits.
    """
    return int(args.batch_size) if args.batch_size else "auto"


def max_batch_for(args: argparse.Namespace) -> int:
    return int(args.batch_size) if args.batch_size else DEFAULT_BATCH.get(args.model, 4)


# ----------------------------------------------------------------- audit ---

def audit_command(args: argparse.Namespace) -> None:
    configure(args.model)
    base.setup_logging(WORKDIR, f"e22-audit-{args.model}")
    model = e19.load_model_fp32(WORKDIR)
    audit = e19.architecture_audit(model)
    audit["kv_site_probe"] = e19.kv_site_probe(model)
    audit["untied_from_tied_checkpoint"] = bool(
        __import__("transformers").AutoConfig.from_pretrained(
            e19.MODEL_ID, cache_dir=str(WORKDIR / "cache" / "huggingface")).tie_word_embeddings)
    audit["compute_dtype"] = "float32"
    audit["git_commit"] = e19.git_commit()
    audit["hardware"] = base.hardware_info()
    base.atomic_json(result_dir(args.model) / f"{PREFIX}_architecture_audit.json", audit)
    LOG.info("E22 audit %s: problems=%s untied=%s", args.model, audit["problems"],
             audit["untied_from_tied_checkpoint"])


# ------------------------------------------------------------ calibration ---

def calibrate_command(args: argparse.Namespace) -> None:
    """R1 (k=8 and k=max) and per-layer R2 through E14; R4 through E18's builder.

    E19 reused E18 v2's frozen R4 factors for the 8B. The other sizes have
    none, so the same streamed randomized-eigenspace builder runs here and
    writes to the layout E19's rotation set reads
    (``activations/<model>/e18v2_factors/<nar_k8|nar_kmax>/down_layer_XX.pt``).
    """
    configure(args.model)
    base.setup_logging(WORKDIR, f"e22-calibrate-{args.model}")
    base.seed_everything(args.seed)
    r4_root = WORKDIR / "activations" / args.model / "e18v2_factors"
    if not (r4_root / "DONE.json").exists():
        model = e19.load_model_fp32(WORKDIR)
        layers = int(model.config.num_hidden_layers)
        # E18's sketch collector hooks both sites unconditionally, so both
        # need bases; only the down factors are read (R1 comes from E14).
        dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
        e18_args = argparse.Namespace(
            workdir=str(WORKDIR), seed=args.seed, calibration_sequences=args.calibration_sequences,
            seq_len=args.seq_len, batch_size=1, oversample=16, permutation_stride=32,
        )
        e18.calibrate(e18_args, model, e19.MODEL_ID, args.model, dimensions, layers,
                      ranks=(8, "max"), root_override=r4_root,
                      eigenspace_name=f"{PREFIX}_calibration_eigenspace.csv")
        if not (r4_root / "DONE.json").exists():
            base.atomic_json(r4_root / "DONE.json", {
                "model": args.model, "model_id": e19.MODEL_ID, "site": "down",
                "ranks": ["nar_k8", "nar_kmax"], "builder": "e18_70b.calibrate",
                "calibration_sequences": args.calibration_sequences, "seed": args.seed,
                "compute_dtype": "float32", "git_commit": e19.git_commit(),
            })
        del model
        gc.collect()
        torch.cuda.empty_cache()
    args.model = args.model  # E14 reads args.model as the model key.
    e14.calibrate_rotations(args)


# -------------------------------------------------------------- benchmarks ---

def c4_tokens(model_id: str, seq_len: int, windows: int) -> torch.Tensor:
    """A contiguous token stream from the first C4 validation shard, E14 chunking."""
    path = WORKDIR / "cache" / "tokenized" / f"{base.model_key_from_id(model_id)}-c4-validation0-w{windows}-l{seq_len}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    dataset = load_dataset("allenai/c4", data_files={"validation": C4_SHARD}, split="validation",
                           cache_dir=str(WORKDIR / "cache" / "datasets"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(WORKDIR / "cache" / "huggingface"), use_fast=True)
    ids: list[int] = []
    needed = windows * seq_len
    for row in dataset:
        ids.extend(tokenizer(row["text"] + "\n\n", add_special_tokens=False)["input_ids"])
        if len(ids) >= needed:
            break
    if len(ids) < needed:
        raise RuntimeError(f"C4 shard yields {len(ids)} tokens, need {needed}")
    chunks = torch.tensor(ids[:needed], dtype=torch.long).reshape(windows, seq_len)
    base.atomic_torch_save(path, chunks)
    return chunks


def _clear_generation_limits(model: torch.nn.Module) -> torch.nn.Module:
    """Let the harness's max_gen_toks govern generation length.

    The Qwen3 checkpoints ship generation_config.max_new_tokens = 2048, and
    transformers lets that override the max_length the harness derives from
    max_gen_toks ("max_new_tokens will take precedence"). The gate's batch-8
    run therefore averaged 1,909 decode steps per batch against a 512 cap.
    """
    cfg = getattr(model, "generation_config", None)
    if cfg is not None:
        cfg.max_new_tokens = None
        cfg.max_length = None
    return model


def build_row(args: argparse.Namespace, row: str) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """The model for one row with its hooks installed, and its provenance."""
    rotation = ROWS[row]
    provenance: dict[str, Any] = {
        "model": args.model, "model_id": e19.MODEL_ID, "row": row, "rotation_checkpoint": rotation,
        "seed": args.seed, "compute_dtype": "float32", "fold_dtype": "float32",
        "gptq_protocol": PROTOCOL if rotation else None, "activation_kind": ACTIVATION_KIND if rotation else None,
        "kv_policy": "E14 KIVI" if rotation else None,
        "git_commit": e19.git_commit(), "hardware": base.hardware_info(),
        "effective_bits": e19.effective_bits(ACTIVATION_KIND if rotation else None, quantize_weights=bool(rotation),
                                             quantize_kv=bool(rotation), shapes=e19.EXPECTED, context=args.seq_len),
    }
    if rotation:
        provenance["effective_bits"]["weight"] = e14.weight_effective_bits(PROTOCOL)
    if rotation is None:
        return _clear_generation_limits(e19.load_model_fp32(WORKDIR)), None, provenance
    model, rotations = e14.load_quantized_model(
        WORKDIR, artifact_root(), args.model, rotation, args.seed, args.weight_row_batch, protocol=PROTOCOL)
    _clear_generation_limits(model)
    done = json.loads((e14.checkpoint_dir(artifact_root(), args.model, rotation, args.seed, PROTOCOL) / "DONE.json").read_text())
    provenance["weight_fold_audit"] = done["fold"]
    provenance["gptq"] = done["gptq"]
    trip = e19.round_trip_audit(rotations, rotations.layers, tolerance=args.round_trip_tolerance)
    provenance["round_trip_max_relative_error"] = max(e["round_trip_relative_error"] for e in trip)
    provenance["ranks"] = rotations.ranks()
    hooks = e14.RuntimeHooks(model, rotations, activation_kind=ACTIVATION_KIND, quantize_kv=True)
    hooks.install()
    return model, hooks, provenance


def harness_config(benchmark: str, model_key: str) -> dict[str, Any]:
    spec = dict(BENCHMARKS[benchmark])
    if benchmark == "math" and model_key in MATH_BENCHMARK:
        spec = dict(BENCHMARKS[MATH_BENCHMARK[model_key]])
    return spec


def run_harness(model: torch.nn.Module, args: argparse.Namespace, spec: dict[str, Any],
                limit: int | None = None, log_samples: bool = False) -> dict[str, Any]:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(e19.MODEL_ID, cache_dir=str(WORKDIR / "cache" / "huggingface"), use_fast=True)
    batch = batch_size_for(args)
    cap = max_batch_for(args)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch, max_batch_size=cap,
              max_length=args.max_length)
    kwargs: dict[str, Any] = {}
    if spec.get("num_fewshot") is not None:
        kwargs["num_fewshot"] = spec["num_fewshot"]
    if spec.get("gen_kwargs"):
        kwargs["gen_kwargs"] = spec["gen_kwargs"]
    if limit is not None:
        kwargs["limit"] = limit
    result = lm_eval.simple_evaluate(
        model=lm, tasks=list(spec["tasks"]), batch_size=batch, max_batch_size=cap,
        task_manager=TaskManager(), cache_requests=False, bootstrap_iters=0,
        log_samples=log_samples, random_seed=args.seed, numpy_random_seed=args.seed,
        torch_random_seed=args.seed, fewshot_random_seed=args.seed,
        apply_chat_template=False, fewshot_as_multiturn=False, **kwargs,
    )
    if result is None:
        raise RuntimeError("lm-eval returned no result")
    del lm
    return result


def headline(spec: dict[str, Any], results: dict[str, Any]) -> tuple[str, float]:
    name, metric = spec["headline"]
    if name == "__mean__":
        values = [float(results[t][e14.ALL_METRICS[t]]) for t in spec["tasks"]]
        return metric, 100.0 * float(np.mean(values))
    table = results.get(name) or {}
    if metric in table:
        return f"{name}:{metric}", 100.0 * float(table[metric])
    # Fall back to the first exact_match / acc key the task reports.
    for key, value in table.items():
        if key.startswith(("exact_match", "acc")) and "stderr" not in key and isinstance(value, (int, float)):
            return f"{name}:{key}", 100.0 * float(value)
    raise KeyError(f"no headline metric for {name} in {list(table)}")


def evaluate_command(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E22 evaluation requires CUDA")
    configure(args.model)
    benchmark = args.benchmark
    if benchmark == "math" and args.model in MATH_BENCHMARK:
        benchmark_file = MATH_BENCHMARK[args.model]
    else:
        benchmark_file = benchmark
    path = result_dir(args.model) / f"{PREFIX}_{args.row}_{benchmark_file}.json"
    if path.exists():
        LOG.info("E22 row exists: %s", path)
        return
    if benchmark == "gpqa":
        from huggingface_hub import get_token
        if not get_token():
            raise RuntimeError("GPQA is a gated dataset; no Hub token is present")
    base.setup_logging(WORKDIR, f"e22-evaluate-{args.model}-{args.row}-{benchmark}")
    base.seed_everything(args.seed)
    started = time.time()
    model, hooks, provenance = build_row(args, args.row)
    try:
        spec = harness_config(benchmark, args.model)
        if spec["kind"] == "ppl":
            tokens = (e19.eval_tokens(WORKDIR, args.seq_len) if benchmark == "wikitext"
                      else c4_tokens(e19.MODEL_ID, args.seq_len, C4_WINDOWS))
            ppl, rows = e19.evaluate_ppl_fp32(model, tokens, f"{args.model} {args.row} {benchmark}")
            payload = {**provenance, "benchmark": benchmark, "ppl": ppl, "chunks": rows,
                       "chunks_evaluated": len(rows), "sequence_length": args.seq_len,
                       "dataset": ("WikiText-2 raw test, first 146 windows at context 2048"
                                   if benchmark == "wikitext" else
                                   f"allenai/c4 {C4_SHARD}, documents joined with blank lines, "
                                   f"first {C4_WINDOWS} windows at context {args.seq_len}"),
                       "nll_dtype": "float32", "headline_metric": "ppl", "headline": ppl}
        else:
            result = run_harness(model, args, spec)
            metric_name, value = headline(spec, result["results"])
            payload = {**provenance, "benchmark": benchmark, "tasks": spec["tasks"],
                       "num_fewshot": spec.get("num_fewshot"), "gen_kwargs": spec.get("gen_kwargs"),
                       "batch_size": batch_size_for(args), "max_batch_size": max_batch_for(args),
                       "max_length": args.max_length,
                       "harness_commit": e14.HARNESS_COMMIT,
                       "results": e14._serializable(result["results"]),
                       "versions": e14._serializable(result.get("versions", {})),
                       "sample_counts": e14._serializable(result.get("n-samples", {})),
                       "headline_metric": metric_name, "headline": value}
            anchor = TECH_REPORT.get(args.model, {}).get(benchmark.replace("math500", "math"))
            payload["tech_report_16bit"] = anchor
        payload["elapsed_seconds"] = time.time() - started
        base.atomic_json(path, payload)
        LOG.info("E22 %s %s %s: %s = %s", args.model, args.row, benchmark,
                 payload["headline_metric"], payload["headline"])
    finally:
        if hooks is not None:
            hooks.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()


# ----------------------------------------------------------------- GPTQ ---

def gptq_command(args: argparse.Namespace) -> None:
    configure(args.model)
    args.protocol = PROTOCOL
    args.artifact_root = str(artifact_root())
    e14.gptq_quantize(args)


def control_command(args: argparse.Namespace) -> None:
    configure(args.model)
    args.rotations = list(args.rotations or ROTATIONS)
    e19.control_command(args)


# ------------------------------------------------------- generation gate ---

class HookCounter:
    """Wrap the KV attention hook to record what it does during decoding."""

    def __init__(self, hooks: e14.RuntimeHooks):
        self.inner = hooks.attention
        self.calls = 0
        self.decode_calls = 0
        self.prefill_calls = 0
        self.k_tokens_quantized = 0
        self.v_pairs_quantized = 0
        self.padded_batches = 0
        self.first_decode: dict[str, Any] | None = None
        hooks.attention = self  # registered by install()

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        q_len, kv_len = int(query.shape[-2]), int(key.shape[-2])
        self.calls += 1
        prefix = (max(0, kv_len - 1) // e14.KV_RESIDUAL_LENGTH) * e14.KV_RESIDUAL_LENGTH
        valid = e14._key_validity(attention_mask, key.shape[0], kv_len)
        if valid is not None:
            self.padded_batches += 1
        if q_len == 1:
            self.decode_calls += 1
            self.k_tokens_quantized += prefix * int(key.shape[0])
            self.v_pairs_quantized += max(0, kv_len - e14.KV_RESIDUAL_LENGTH) * int(key.shape[0])
            if self.first_decode is None:
                self.first_decode = {"kv_length": kv_len, "k_prefix_quantized": prefix,
                                     "v_tokens_quantized_for_this_query": max(0, kv_len - e14.KV_RESIDUAL_LENGTH),
                                     "batch": int(key.shape[0]), "padded": valid is not None}
        else:
            self.prefill_calls += 1
        return self.inner(module, query, key, value, attention_mask, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("calls", "decode_calls", "prefill_calls", "k_tokens_quantized",
                                              "v_pairs_quantized", "padded_batches", "first_decode")}


def _samples(result: dict[str, Any], task: str) -> list[str]:
    return [str(s["resps"][0][0]) for s in result["samples"][task]]


def gate_command(args: argparse.Namespace) -> None:
    """Before any generative benchmark: does decoding go through the quantizers?

    100 GSM8K items through the NAR k=max W4A4KV4 row with generate(): (a) the
    KV hook fires on every decode step and quantizes completed K chunks and
    older V per the PPL policy; (b) greedy decoding is deterministic (two runs
    identical); (c) batched decoding agrees with batch-1 decoding, the only
    place the padding path matters; (d) the fp32-container bf16 row reproduces
    the harness's stock bf16 run within noise.
    """
    configure(args.model)
    base.setup_logging(WORKDIR, f"e22-gate-{args.model}")
    spec = dict(BENCHMARKS["gsm8k"])
    limit = args.items
    out: dict[str, Any] = {"model": args.model, "model_id": e19.MODEL_ID, "items": limit,
                           "task": spec["tasks"], "num_fewshot": spec["num_fewshot"],
                           "git_commit": e19.git_commit(), "hardware": base.hardware_info()}
    task = spec["tasks"][0]

    model, hooks, provenance = build_row(args, "nar_kmax_asym_g128")
    counter = HookCounter(hooks)
    hooks.close()
    hooks.install()  # re-register with the counting wrapper
    try:
        runs = []
        for label, batch in (("batch1_run1", 1), ("batch1_run2", 1), (f"batch{args.gate_batch}", args.gate_batch)):
            args.batch_size = batch
            before = counter.summary()
            result = run_harness(model, args, spec, limit=limit, log_samples=True)
            after = counter.summary()
            metric, value = headline(spec, result["results"])
            runs.append({"label": label, "batch_size": batch, "headline_metric": metric, "accuracy": value,
                         "hook_activity": {k: (after[k] - before[k]) if isinstance(after[k], int) else after[k]
                                           for k in after},
                         "responses": _samples(result, task)})
            LOG.info("gate %s: %s = %.2f, hooks %s", label, metric, value, runs[-1]["hook_activity"])
    finally:
        hooks.close()
        counter.inner = None  # the counter held the hooks, the hooks held the model
        del counter, hooks, model
        gc.collect()
        torch.cuda.empty_cache()
    r1, r2, rb = runs
    out["quantized_row"] = {
        "provenance": provenance,
        "runs": [{k: v for k, v in r.items() if k != "responses"} for r in runs],
        "decode_hook_fires": r1["hook_activity"]["decode_calls"] > 0,
        "k_quantized_during_decode": r1["hook_activity"]["k_tokens_quantized"] > 0,
        "v_quantized_during_decode": r1["hook_activity"]["v_pairs_quantized"] > 0,
        "deterministic": r1["responses"] == r2["responses"],
        "batched_agreement": float(np.mean([a == b for a, b in zip(r1["responses"], rb["responses"])])),
        "batched_padding_seen": rb["hook_activity"]["padded_batches"] > 0,
    }

    # (d) the fp32-container bf16 row against a plain bf16 load through the same harness.
    args.batch_size = 1
    model = e19.load_model_fp32(WORKDIR)
    try:
        result = run_harness(model, args, spec, limit=limit, log_samples=True)
        fp32_acc = headline(spec, result["results"])[1]
        fp32_resps = _samples(result, task)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM
    stock = AutoModelForCausalLM.from_pretrained(
        e19.MODEL_ID, cache_dir=str(WORKDIR / "cache" / "huggingface"), dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval().cuda()
    _clear_generation_limits(stock)
    try:
        result = run_harness(stock, args, spec, limit=limit, log_samples=True)
        stock_acc = headline(spec, result["results"])[1]
        stock_resps = _samples(result, task)
    finally:
        del stock
        gc.collect()
        torch.cuda.empty_cache()
    out["bf16_row"] = {"fp32_container_accuracy": fp32_acc, "stock_bf16_accuracy": stock_acc,
                       "difference_points": fp32_acc - stock_acc,
                       "response_agreement": float(np.mean([a == b for a, b in zip(fp32_resps, stock_resps)])),
                       "within_noise_1pt": abs(fp32_acc - stock_acc) <= max(1.0, 100.0 / math.sqrt(limit))}
    q = out["quantized_row"]
    out["passed"] = bool(q["decode_hook_fires"] and q["k_quantized_during_decode"] and q["v_quantized_during_decode"]
                         and q["deterministic"] and out["bf16_row"]["within_noise_1pt"])
    out["batched_agreement_note"] = ("batch-1 and batched decoding differ only through the padded positions' "
                                     "chunk phase; the agreement rate is reported, not gated")
    base.atomic_json(result_dir(args.model) / f"{PREFIX}_generation_gate.json", out)
    base.atomic_json(result_dir(args.model) / f"{PREFIX}_generation_gate_responses.json",
                     {"runs": runs, "fp32_bf16_row": fp32_resps, "stock_bf16": stock_resps})
    LOG.info("E22 generation gate passed=%s: %s", out["passed"], json.dumps({k: v for k, v in out.items() if k != "quantized_row"}, indent=1, default=str)[:2000])
    if not out["passed"]:
        raise AssertionError("E22 generation gate failed")


# -------------------------------------------------------------- finalize ---

def benchmark_config_command(args: argparse.Namespace) -> None:
    """Write the resolved harness configuration of every benchmark to one file."""
    import lm_eval
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    out: dict[str, Any] = {"harness_commit": e14.HARNESS_COMMIT, "lm_eval_version": lm_eval.__version__,
                           "benchmarks": {}}
    for name, spec in BENCHMARKS.items():
        entry: dict[str, Any] = {k: v for k, v in spec.items()}
        if spec["kind"] == "harness":
            entry["task_configs"] = {}
            for task_name in spec["tasks"]:
                try:
                    d = lm_eval.tasks.get_task_dict([task_name], tm)
                except Exception as error:  # gated datasets
                    entry["task_configs"][task_name] = {"error": str(error)[:200]}
                    continue
                leaf = d  # group tasks come back keyed by group objects; descend to a leaf
                while isinstance(leaf, dict):
                    leaf = list(leaf.values())[0]
                cfg = leaf.config.to_dict() if hasattr(leaf.config, "to_dict") else dict(leaf.config.__dict__)
                entry["task_configs"][task_name] = e14._serializable({
                    k: cfg.get(k) for k in ("task", "dataset_path", "dataset_name", "num_fewshot", "output_type",
                                            "generation_kwargs", "filter_list", "metric_list", "doc_to_text",
                                            "fewshot_config") if k in cfg})
                entry["task_configs"][task_name]["representative_leaf"] = getattr(leaf.config, "task", task_name)
        out["benchmarks"][name] = entry
    out["notes"] = {
        "mmlu_redux": "the pinned harness has MMLU-Redux only in its generative form; max_gen_toks 8 and a newline stop, first letter extracted",
        "gsm8k": "gsm8k_cot carries eight fixed CoT exemplars with a first_n sampler; num_fewshot=4 takes the first four; flexible-extract is the headline, strict-match is recorded; max_gen_toks 512 against the harness default of 2048",
        "bbh": "bbh_cot_fewshot is 3-shot CoT by construction; the task's own max_gen_toks 1024 and stop strings are kept",
        "math": "minerva_math500 (4 fixed CoT exemplars) at every size; exact_match is the headline, math_verify is recorded",
        "bbh": "defined but not run: dropped on 2026-09-07 in favour of the hellaswag supplementary task",
        "hellaswag": "zero-shot acc_norm, the supplementary task replacing BBH; the eight-task zero-shot set is run on the 8B only",
        "gpqa": "gpqa_main_cot_n_shot at 5 shots; gated dataset, needs a Hub token with access",
        "generation": "greedy (do_sample false) throughout; the KV cache is quantized on every decode step by the E14 attention hook",
        "padding": "batched decoding pads on the left; padded keys are replaced by their nearest valid neighbour before chunk statistics, and the chunk phase of padded sequences shifts by the pad length; measured by the generation gate",
    }
    path = Path(args.output)
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    LOG.info("wrote %s", path)


def finalize_command(args: argparse.Namespace) -> None:
    """Collect every E22 artifact into results/e22_family_summary.csv."""
    rows: list[dict[str, Any]] = []
    for model_key in FAMILY:
        directory = WORKDIR / "results" / model_key
        if not directory.exists():
            continue
        for path in sorted(directory.glob(f"{PREFIX}_*_*.json")):
            payload = json.loads(path.read_text())
            if "headline" not in payload or "row" not in payload:
                continue
            bench = payload["benchmark"]
            anchor = payload.get("tech_report_16bit")
            row = {
                "model": model_key, "row": payload["row"], "benchmark": bench,
                "headline_metric": payload["headline_metric"], "value": payload["headline"],
                "tech_report_16bit": anchor,
                "bf16_minus_report": None, "flag_pipeline_difference": None,
                "effective_bits": json.dumps(payload.get("effective_bits")),
                "gptq_protocol": payload.get("gptq_protocol"), "git_commit": payload.get("git_commit"),
                "elapsed_seconds": payload.get("elapsed_seconds"), "artifact": str(path.relative_to(WORKDIR)),
            }
            if payload["row"] == "bf16" and anchor is not None:
                row["bf16_minus_report"] = payload["headline"] - anchor
                row["flag_pipeline_difference"] = abs(row["bf16_minus_report"]) > ANCHOR_FLAG_POINTS
            rows.append(row)
    base.write_csv(WORKDIR / "results" / f"{PREFIX}_family_summary.csv", rows)
    LOG.info("E22 summary: %d rows", len(rows))
    for r in rows:
        print(f"{r['model']:18s} {r['row']:22s} {r['benchmark']:10s} {r['value']:8.3f}  report={r['tech_report_16bit']}  flag={r['flag_pipeline_difference']}")


# ------------------------------------------------------------------- CLI ---

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--model", choices=FAMILY, default="qwen3_8b_base")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--seq-len", type=int, default=2048)
    result.add_argument("--max-length", type=int, default=4096, help="harness context limit for few-shot prompts")
    result.add_argument("--weight-row-batch", type=int, default=256)
    result.add_argument("--calibration-sequences", type=int, default=128)
    result.add_argument("--calibration-seed", type=int, default=0)
    result.add_argument("--verify-tokens", type=int, default=128)
    result.add_argument("--fold-tolerance", type=float, default=1e-2)
    result.add_argument("--round-trip-tolerance", type=float, default=1e-6)
    result.add_argument("--batch-size", type=int, default=0, help="0 = per-model default")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("calibrate")
    control = sub.add_parser("control")
    control.add_argument("--rotations", nargs="+", choices=ROTATIONS)
    control.add_argument("--control-chunks", type=int, default=64)
    control.add_argument("--control-ppl-tolerance", type=float, default=0.01)
    control.add_argument("--control-nll-tolerance", type=float, default=1e-3)
    gptq = sub.add_parser("gptq")
    gptq.add_argument("--rotation", choices=ROTATIONS, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--row", choices=tuple(ROWS), required=True)
    evaluate.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--items", type=int, default=100)
    gate.add_argument("--gate-batch", type=int, default=8)
    config = sub.add_parser("benchmark-config")
    config.add_argument("--output", default=str(Path(__file__).resolve().parent / "e22_benchmarks.json"))
    sub.add_parser("finalize")
    return result


def main() -> None:
    global WORKDIR
    args = parser().parse_args()
    WORKDIR = Path(args.workdir).resolve()
    args.artifact_root = str(artifact_root())
    {
        "audit": audit_command, "calibrate": calibrate_command, "control": control_command,
        "gptq": gptq_command, "evaluate": evaluate_command, "gate": gate_command,
        "benchmark-config": benchmark_config_command, "finalize": finalize_command,
    }[args.command](args)


if __name__ == "__main__":
    main()
