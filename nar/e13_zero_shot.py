#!/usr/bin/env python3
"""E13 zero-shot transfer check using a pinned lm-evaluation-harness."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
except ImportError:
    import activation_experiments as act
    import experiment as base


MODELS = ("llama32_3b", "llama31_8b")
METHODS = ("bf16", "hadamard", "nar")
TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "lambada_openai")
METRICS = {
    "piqa": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "lambada_openai": "acc,none",
}
HARNESS_COMMIT = "b954108c9baaaa934b4ad842033b31a97ee30816"


def _serializable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda item: item.item() if hasattr(item, "item") else str(item)))


def result_path(workdir: Path, model: str, method: str) -> Path:
    return workdir / "results" / model / f"e13_{method}.json"


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E13 requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    asset_workdir = Path(args.asset_workdir).resolve() if args.asset_workdir else workdir
    output = result_path(workdir, args.model, args.method)
    if output.exists():
        print(f"E13 result exists: {output}")
        return
    base.setup_logging(workdir, f"e13-{args.model}-{args.method}")
    base.seed_everything(args.seed)
    from transformers import AutoTokenizer
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    model_id, model_key = act.model_id_and_key(args.model)
    model = base.load_model(model_id, asset_workdir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=str(asset_workdir / "cache" / "huggingface"), use_fast=True
    )
    hooks = None
    fold_error = 0.0
    if args.method != "bf16":
        layers = int(model.config.num_hidden_layers)
        dimensions = {"qkv": int(model.config.hidden_size), "down": int(model.config.intermediate_size)}
        rotations = act.MethodRotations(
            workdir, model_key, args.method, 0, args.seed, layers, dimensions, torch.device("cuda")
        )
        weights = act.WeightManager(model)
        q_error = weights.rotate("qkv", rotations, args.weight_row_batch)
        d_error = weights.rotate("down", rotations, args.weight_row_batch)
        fold_error = max(q_error, d_error)
        hooks = act.ActivationQuantHooks(model, rotations, True, True)
        hooks.install()
    lm = HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size,
        max_batch_size=args.batch_size, max_length=args.max_length,
    )
    task_manager = TaskManager()
    try:
        result = lm_eval.simple_evaluate(
            model=lm, tasks=list(TASKS), num_fewshot=0, batch_size=args.batch_size,
            max_batch_size=args.batch_size, task_manager=task_manager,
            cache_requests=False, bootstrap_iters=0, log_samples=False,
            limit=args.limit,
            random_seed=args.seed, numpy_random_seed=args.seed,
            torch_random_seed=args.seed, fewshot_random_seed=args.seed,
            apply_chat_template=False, fewshot_as_multiturn=False,
        )
    finally:
        if hooks is not None:
            hooks.close()
    if result is None:
        raise RuntimeError("lm-eval returned no rank-0 result")
    rows = []
    for task in TASKS:
        metric = METRICS[task]
        accuracy = float(result["results"][task][metric])
        rows.append({"task": task, "metric": metric, "accuracy": accuracy})
    mean_accuracy = float(np.mean([row["accuracy"] for row in rows]))
    base.atomic_json(output, {
        "model": model_key, "model_id": model_id, "method": args.method,
        "activation_site": "both", "activation_quantizer": (
            "none" if args.method == "bf16" else "dynamic asymmetric per-token group-128 INT4"
        ),
        "seed": args.seed, "num_fewshot": 0, "tasks": rows,
        "mean_accuracy": mean_accuracy, "mean_definition": "unweighted mean of the six selected accuracy metrics",
        "harness_commit": HARNESS_COMMIT, "batch_size": args.batch_size,
        "max_length": args.max_length, "smoke_limit": args.limit,
        "weight_fold_max_relative_error": fold_error,
        "raw_results": _serializable(result.get("results", {})),
        "task_versions": _serializable(result.get("versions", {})),
        "sample_counts": _serializable(result.get("n-samples", {})),
        "hardware": base.hardware_info(),
    })
    del lm, model
    gc.collect()
    torch.cuda.empty_cache()


def finalize(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        path = result_path(workdir, args.model, method)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        payloads[method] = payload
        for row in payload["tasks"]:
            rows.append({
                "model": args.model, "method": method, "task": row["task"],
                "metric": row["metric"], "accuracy": row["accuracy"],
                "delta_vs_bf16": float(row["accuracy"]) - next(
                    float(item["accuracy"]) for item in payloads["bf16"]["tasks"] if item["task"] == row["task"]
                ),
                "seed": payload["seed"],
            })
        rows.append({
            "model": args.model, "method": method, "task": "mean",
            "metric": "unweighted_mean_selected_accuracy", "accuracy": payload["mean_accuracy"],
            "delta_vs_bf16": payload["mean_accuracy"] - payloads["bf16"]["mean_accuracy"],
            "seed": payload["seed"],
        })
    result_dir = workdir / "results" / args.model
    base.write_csv(result_dir / "e13_zero_shot.csv", rows)
    base.atomic_json(result_dir / "E13_DONE.json", {
        "model": args.model, "methods": list(METHODS), "tasks": list(TASKS),
        "metrics": METRICS, "seed": args.seed, "num_fewshot": 0,
        "paired": "same harness revision, task examples, prompts, metrics, and seed for every method",
        "harness_commit": HARNESS_COMMIT, "no_tuning": True,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--asset-workdir")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model", choices=MODELS, required=True)
    run_parser.add_argument("--method", choices=METHODS, required=True)
    run_parser.add_argument("--batch-size", type=int, default=2)
    run_parser.add_argument("--max-length", type=int, default=2048)
    run_parser.add_argument("--weight-row-batch", type=int, default=512)
    run_parser.add_argument("--limit", type=float)
    final_parser = sub.add_parser("finalize")
    final_parser.add_argument("--model", choices=MODELS, required=True)
    return parser


if __name__ == "__main__":
    parsed = parser().parse_args()
    {"run": run, "finalize": finalize}[parsed.command](parsed)
