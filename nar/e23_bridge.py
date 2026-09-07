"""E23 — bridge to the published Qwen3 W4A4 tables (MosaicQuant, TwinQuant).

MosaicQuant (arXiv 2606.15652) and TwinQuant (arXiv 2606.01556) report the
same 16-bit rows (Llama-3.2-3B 10.7, Llama-3-8B 8.6, Qwen3-8B 9.71, 14B 8.6,
32B 7.6 on WikiText-2; identical six-task per-task accuracies), so they share
one evaluation protocol.  Neither paper states the checkpoints, the
perplexity context length or chunking, or the harness version, and neither
has released code.  This module therefore

  gate      evaluates candidate checkpoints under candidate perplexity
            conventions and the six tasks, and reports which combination
            reproduces the published 16-bit row within 0.1 PPL;
  (later)   runs our rows under the identified protocol.

Published rows are held in PUBLISHED verbatim from the papers' appendix
tables (Table 3/4 in MosaicQuant, Table 4/5 in TwinQuant).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nar import e14_w4a4kv4 as e14  # noqa: E402
from nar import experiment as base  # noqa: E402

LOG = logging.getLogger("nar.e23")
WORKDIR = Path(".")

SIX_TASKS = ("arc_challenge", "arc_easy", "hellaswag", "lambada_openai", "piqa", "winogrande")
TASK_COLUMNS = ("ARC-C", "ARC-E", "HellaSwag", "PIQA", "Winogrande", "LAMBADA")

# 16-bit rows as printed (MosaicQuant Tables 3–4, TwinQuant Tables 4–5; the
# two papers agree on every entry below except the Qwen3-4B row, which is
# 10.04 / 69.4 in MosaicQuant and 13.7 / 66.8 in TwinQuant).
PUBLISHED_16BIT = {
    "llama3_3b": {"wiki2": 10.7, "avg": 66.2, "ARC-C": 47.6, "ARC-E": 69.9, "HellaSwag": 71.0, "PIQA": 76.0, "Winogrande": 66.6, "LAMBADA": 65.9},
    "llama3_8b": {"wiki2": 8.64, "avg": 73.5, "ARC-C": 54.86, "ARC-E": 79.55, "HellaSwag": 79.13, "PIQA": 80.74, "Winogrande": 73.72, "LAMBADA": 72.93},
    "qwen3_4b_mosaic": {"wiki2": 10.04, "avg": 69.4, "ARC-C": 58.6, "ARC-E": 81.1, "HellaSwag": 69.09, "PIQA": 76.0, "Winogrande": 68.11, "LAMBADA": 63.56},
    "qwen3_4b_twin": {"wiki2": 13.7, "avg": 66.8, "ARC-C": 50.6, "ARC-E": 80.5, "HellaSwag": 69.5, "PIQA": 75.0, "Winogrande": 65.8, "LAMBADA": 59.4},
    "qwen3_8b": {"wiki2": 9.71, "avg": 71.6, "ARC-C": 55.5, "ARC-E": 83.5, "HellaSwag": 78.8, "PIQA": 76.4, "Winogrande": 68.0, "LAMBADA": 67.4},
    "qwen3_14b": {"wiki2": 8.6, "avg": 74.2, "ARC-C": 59.0, "ARC-E": 84.3, "HellaSwag": 80.5, "PIQA": 80.0, "Winogrande": 72.9, "LAMBADA": 68.4},
    "qwen3_32b": {"wiki2": 7.6, "avg": 75.2, "ARC-C": 57.8, "ARC-E": 84.4, "HellaSwag": 84.2, "PIQA": 80.9, "Winogrande": 73.6, "LAMBADA": 70.3},
}

# Candidate checkpoints for each published row.
CANDIDATES = {
    "llama3_3b": ["unsloth/Llama-3.2-3B", "unsloth/Llama-3.2-3B-Instruct"],
    "llama3_8b": ["unsloth/llama-3-8b", "unsloth/Meta-Llama-3.1-8B"],
    "qwen3_4b": ["Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B"],
    "qwen3_8b": ["Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"],
    "qwen3_14b": ["Qwen/Qwen3-14B-Base", "Qwen/Qwen3-14B"],
    "qwen3_32b": ["Qwen/Qwen3-32B"],
}

CONTEXTS = (2048, 4096)


def result_dir() -> Path:
    d = WORKDIR / "results" / "e23"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_bf16(model_id: str) -> tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = str(WORKDIR / "cache" / "huggingface")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache, dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="sdpa")
    model.eval()
    model.generation_config.max_new_tokens = None
    model.generation_config.max_length = None
    return model, tokenizer


def wikitext_stream(model_id: str, tokenizer: Any, bos: str) -> torch.Tensor:
    """The WikiText-2 raw test set as one token stream.

    bos="start": one BOS token at the start of the stream (QuaRot / GPTQ
    convention: tokenizer("\\n\\n".join(test)) with special tokens);
    bos="none": no BOS anywhere.
    """
    from datasets import load_dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                           cache_dir=str(WORKDIR / "cache" / "datasets"))
    text = "\n\n".join(row["text"] for row in dataset)
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if bos == "start":
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            from transformers import AutoConfig
            bos_id = getattr(AutoConfig.from_pretrained(model_id, cache_dir=str(WORKDIR / "cache" / "huggingface")),
                             "bos_token_id", None)
        if bos_id is not None:
            ids = [bos_id] + ids
    return torch.tensor(ids, dtype=torch.long)


@torch.inference_mode()
def windowed_ppl(model: torch.nn.Module, stream: torch.Tensor, seq_len: int, limit: int | None,
                 label: str) -> dict[str, Any]:
    """Non-overlapping windows of seq_len over the stream, token-level PPL
    (weighted by scored tokens), fp32 NLL.  This is the QuaRot/GPTQ
    `eval_ppl` convention when the stream carries one leading BOS."""
    n = stream.numel() // seq_len
    if limit is not None:
        n = min(n, limit)
    device = next(model.parameters()).device
    nll_sum, count = 0.0, 0
    for i in range(n):
        batch = stream[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device)
        logits = model(input_ids=batch, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum")
        nll_sum += float(loss); count += int(batch.shape[1] - 1)
        if i % 32 == 0:
            LOG.info("%s L=%d window %d/%d running ppl=%.4f", label, seq_len, i + 1, n, math.exp(nll_sum / count))
        del logits, loss
    return {"seq_len": seq_len, "windows": n, "tokens_scored": count, "ppl": math.exp(nll_sum / count)}


def our_windows_ppl(model: torch.nn.Module, model_id: str, tokenizer: Any, seq_len: int,
                    limit: int | None) -> dict[str, Any]:
    """This repository's convention: every window is BOS + (seq_len-1) content
    tokens, contiguous content, all windows (E14 uses the first 141/146)."""
    content = wikitext_stream(model_id, tokenizer, "none")
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        from transformers import AutoConfig
        bos_id = getattr(AutoConfig.from_pretrained(model_id, cache_dir=str(WORKDIR / "cache" / "huggingface")), "bos_token_id", None)
    content_len = seq_len - 1
    n = content.numel() // content_len
    if limit is not None:
        n = min(n, limit)
    device = next(model.parameters()).device
    nll_sum, count = 0.0, 0
    with torch.inference_mode():
        for i in range(n):
            window = content[i * content_len:(i + 1) * content_len]
            if bos_id is not None:
                window = torch.cat([torch.tensor([bos_id]), window])
            batch = window.unsqueeze(0).to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum")
            nll_sum += float(loss); count += int(batch.shape[1] - 1)
            del logits, loss
    return {"seq_len": seq_len, "windows": n, "tokens_scored": count, "ppl": math.exp(nll_sum / count),
            "convention": "BOS per window (E14/E19/E22), all windows"}


def git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def run_harness(model: torch.nn.Module, tokenizer: Any, tasks: list[str], max_length: int,
                limit: int | None, batch: str | int = "auto") -> dict[str, Any]:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch, max_batch_size=32, max_length=max_length)
    kwargs: dict[str, Any] = {}
    if limit is not None:
        kwargs["limit"] = limit
    out = lm_eval.simple_evaluate(model=lm, tasks=tasks, batch_size=batch, max_batch_size=32,
                                  task_manager=TaskManager(), cache_requests=False, bootstrap_iters=0,
                                  apply_chat_template=False, **kwargs)
    del lm
    return out["results"]


def six_task_table(results: dict[str, Any]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for task in SIX_TASKS:
        r = results[task]
        table[task] = {k: 100.0 * float(v) for k, v in r.items()
                       if k.split(",")[0] in ("acc", "acc_norm") and "stderr" not in k and isinstance(v, (int, float))}
    acc = [table[t]["acc,none"] for t in SIX_TASKS]
    norm = [table[t].get("acc_norm,none", table[t]["acc,none"]) for t in SIX_TASKS]
    return {"per_task": table, "mean_acc": float(np.mean(acc)), "mean_acc_norm_where_defined": float(np.mean(norm))}


def gate_command(args: argparse.Namespace) -> None:
    model_id = args.model_id
    key = model_id.replace("/", "--")
    path = result_dir() / f"{key}_bf16_gate.json"
    if path.exists() and not args.force:
        LOG.info("E23 gate exists: %s", path)
        return
    base.setup_logging(WORKDIR, f"e23-gate-{key}")
    started = time.time()
    model, tokenizer = load_bf16(model_id)
    out: dict[str, Any] = {"model_id": model_id, "key": key, "compute_dtype": "bfloat16",
                           "hardware": base.hardware_info(), "git_commit": git_commit(),
                           "perplexity": {}, "harness": {}}
    limit = args.limit_windows
    for seq_len in CONTEXTS:
        stream = wikitext_stream(model_id, tokenizer, "start")
        out["perplexity"][f"windows_bos_start_L{seq_len}"] = windowed_ppl(model, stream, seq_len, limit, "bos_start")
        out["perplexity"][f"windows_bos_each_L{seq_len}"] = our_windows_ppl(model, model_id, tokenizer, seq_len, limit)
        base.atomic_json(path.with_suffix(".partial.json"), out)
    for max_length in args.harness_lengths:
        res = run_harness(model, tokenizer, ["wikitext"], max_length, args.limit_docs)
        r = res["wikitext"]
        out["harness"][f"wikitext_rolling_L{max_length}"] = {k: float(v) for k, v in r.items()
                                                             if isinstance(v, (int, float))}
        base.atomic_json(path.with_suffix(".partial.json"), out)
    res = run_harness(model, tokenizer, list(SIX_TASKS), 4096, args.limit_docs)
    out["harness"]["six_task"] = six_task_table(res)
    out["elapsed_seconds"] = time.time() - started
    out["published_rows"] = PUBLISHED_16BIT
    base.atomic_json(path, out)
    path.with_suffix(".partial.json").unlink(missing_ok=True)
    LOG.info("E23 gate %s: %s", key, json.dumps({k: round(v["ppl"], 3) for k, v in out["perplexity"].items()}))
    LOG.info("E23 gate %s harness: %s", key, json.dumps(out["harness"]))


def compare_command(args: argparse.Namespace) -> None:
    """Print every gate artifact against the published rows."""
    rows = []
    for path in sorted(result_dir().glob("*_bf16_gate.json")):
        d = json.loads(path.read_text())
        ppl = {k: round(v["ppl"], 3) for k, v in d["perplexity"].items()}
        wiki = {k: round(v.get("word_perplexity,none", float("nan")), 3) for k, v in d["harness"].items() if k.startswith("wikitext")}
        six = d["harness"]["six_task"]
        print(f"== {d['model_id']}")
        print("   token ppl:", ppl)
        print("   harness word ppl:", wiki)
        print("   six-task acc:", {t: round(v['acc,none'], 1) for t, v in six['per_task'].items()}, "mean", round(six["mean_acc"], 2))
        print("   six-task acc_norm:", {t: round(v.get('acc_norm,none', float('nan')), 1) for t, v in six['per_task'].items()},
              "mean", round(six["mean_acc_norm_where_defined"], 2))
    print("published:", json.dumps(PUBLISHED_16BIT, indent=1))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--model-id", required=True)
    g.add_argument("--limit-windows", type=int, default=None)
    g.add_argument("--limit-docs", type=int, default=None)
    g.add_argument("--harness-lengths", type=int, nargs="+", default=[2048, 4096])
    g.add_argument("--force", action="store_true")
    sub.add_parser("compare")
    return p


def main() -> None:
    global WORKDIR
    args = parser().parse_args()
    WORKDIR = Path(args.workdir)
    {"gate": gate_command, "compare": compare_command}[args.command](args)


if __name__ == "__main__":
    main()
