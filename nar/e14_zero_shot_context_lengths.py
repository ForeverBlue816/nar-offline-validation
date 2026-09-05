"""Prompt-length distribution of the nine zero-shot tasks against the KIVI residual.

Generalises the E19 measurement (`e19_zero_shot_context_lengths.py`, six tasks,
Qwen3 tokenizer) to any E14 model and to the three tasks added for the
published eight-task mean.  BoolQ is the reason: its passages are the longest
prompts in the eight, so it is the one task where most requests actually
quantize the K cache, and the one where NAR k=8 loses the most.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nar import activation_experiments as act  # noqa: E402
from nar import e14_w4a4kv4 as e14  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--model", required=True, choices=tuple(act.MODEL_IDS))
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    model_id, model_key = act.model_id_and_key(args.model)

    import lm_eval
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(workdir / "cache" / "huggingface"), use_fast=True)
    tm = TaskManager()
    R = e14.KV_RESIDUAL_LENGTH
    out = []
    for task_name in e14.TASKS + e14.EXTRA_TASKS:
        task = lm_eval.tasks.get_task_dict([task_name], tm)[task_name]
        while not hasattr(task, "build_all_requests"):
            task = list(task.values())[0]
        task.build_all_requests(limit=None, rank=0, world_size=1,
                                cache_requests=False, tokenizer_name=model_id)
        lens = [len(tok(inst.args[0], add_special_tokens=False)["input_ids"])
                for inst in task.instances if isinstance(inst.args[0], str)]
        a = np.array(lens)
        quantized_k = (np.maximum(0, a - 1) // R) * R
        m = np.maximum(0, a - R)
        out.append({
            "task": task_name, "requests": int(a.size),
            "ctx_min": int(a.min()), "ctx_p50": int(np.percentile(a, 50)),
            "ctx_p90": int(np.percentile(a, 90)), "ctx_max": int(a.max()),
            "ctx_mean": round(float(a.mean()), 2),
            "share_no_kv_quantized_at_all": round(float((a <= R).mean()), 4),
            "quantized_k_share_of_ctx": round(float(quantized_k.sum() / a.sum()), 4),
            "quantized_v_share_of_causal_pairs": round(float((m * (m + 1) / 2).sum() / (a * (a + 1) / 2).sum()), 4),
        })
        print(out[-1], flush=True)
    path = workdir / "results" / model_key / "e14_zero_shot_context_lengths.json"
    path.write_text(json.dumps({
        "model_id": model_id, "kv_residual_length": R,
        "k_rule": "prefix = floor((T-1)/R)*R tokens are quantized; the rest stay bf16",
        "v_rule": "the most recent R tokens are kept bf16 per query position",
        "tasks": out}, indent=2) + "\n")
    print("written", path)


if __name__ == "__main__":
    main()
