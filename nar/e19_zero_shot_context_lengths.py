"""Prompt-length distribution of the six zero-shot tasks against the KIVI residual.

KIVI quantizes K only for completed KV_RESIDUAL_LENGTH-token chunks:
prefix = floor((T-1)/R)*R, so a sequence with T <= R has prefix 0 and no K is
quantized at all.  V keeps the most recent R tokens in full precision per query.
This measures, per task, the request context length the model actually sees.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nar import e14_w4a4kv4 as e14

R = e14.KV_RESIDUAL_LENGTH
import lm_eval
from lm_eval.tasks import TaskManager
from transformers import AutoTokenizer

WORKDIR = Path("/projects/_hdd/nar/nar-validation")
MODEL_ID = "Qwen/Qwen3-8B-Base"
tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(WORKDIR / "cache" / "huggingface"), use_fast=True)
tm = TaskManager()
out = []
for task_name in e14.TASKS:
    task_dict = lm_eval.tasks.get_task_dict([task_name], tm)
    task = task_dict[task_name]
    while not hasattr(task, "build_all_requests"):
        task = list(task.values())[0]
    task.build_all_requests(limit=None, rank=0, world_size=1,
                            cache_requests=False, tokenizer_name=MODEL_ID)
    lens = []
    for inst in task.instances:
        ctx = inst.args[0]
        if isinstance(ctx, str):
            lens.append(len(tok(ctx, add_special_tokens=False)["input_ids"]))
    a = np.array(lens)
    # number of K tokens that are actually quantized for a prompt of length T
    quantized_k = (np.maximum(0, a - 1) // R) * R
    # V: key k is quantized for query q only when k <= q - R, so over the causal
    # pairs of a length-T sequence the quantized share is m(m+1)/2 over T(T+1)/2
    # with m = max(0, T - R).
    m = np.maximum(0, a - R)
    v_quant_pairs = m * (m + 1) / 2
    causal_pairs = a * (a + 1) / 2
    out.append({
        "task": task_name, "requests": int(a.size),
        "ctx_min": int(a.min()), "ctx_p50": int(np.percentile(a, 50)),
        "ctx_p90": int(np.percentile(a, 90)), "ctx_max": int(a.max()),
        "ctx_mean": round(float(a.mean()), 2),
        "share_ctx_le_R": round(float((a <= R).mean()), 4),
        "quantized_k_tokens_mean": round(float(quantized_k.mean()), 2),
        "quantized_k_share_of_ctx": round(float(quantized_k.sum() / a.sum()), 4),
        "quantized_v_share_of_causal_pairs": round(float(v_quant_pairs.sum() / causal_pairs.sum()), 4),
        "share_no_kv_quantized_at_all": round(float((a <= R).mean()), 4),
    })
    print(out[-1], flush=True)
Path("/projects/_hdd/nar/nar-validation/results/qwen3_8b_base/e19_zero_shot_context_lengths.json").write_text(
    json.dumps({"model_id": MODEL_ID, "kv_residual_length": R,
                "k_rule": "prefix = floor((T-1)/R)*R tokens are quantized; the rest stay bf16",
                "v_rule": "the most recent R tokens are kept bf16 per query position",
                "tasks": out}, indent=2) + "\n")
print("written")
