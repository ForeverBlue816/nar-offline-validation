"""Where does batched decoding time go under the quantizer hooks?

Times prefill plus 32 greedy decode steps on padded batches of 1 and 8 for
the stock model, the row with all hooks, KV hook only, and activation hooks
only, on a small family member so weight bandwidth is negligible.
"""
import argparse, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nar import e22_qwen3_family as e22, e19_qwen3_e2e as e19, e14_w4a4kv4 as e14, experiment as base

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", required=True)
parser.add_argument("--model", default="qwen3_0.6b_base")
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--repeat-prompt", type=int, default=4, help="prompt length multiplier")
parser.add_argument("--profile", action="store_true", help="profile one batch-8 decode step under all hooks")
parser.add_argument("--configs", default="stock,all_hooks,kv_only,act_only")
args = parser.parse_args()
e22.WORKDIR = Path(args.workdir).resolve()
args.seed, args.weight_row_batch, args.round_trip_tolerance, args.seq_len = 0, 256, 1e-6, 2048
e22.configure(args.model)
base.setup_logging(e22.WORKDIR, f"e22-decode-timing-{args.model}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(e19.MODEL_ID, cache_dir=str(e22.WORKDIR / "cache" / "huggingface")); tok.padding_side = "left"
prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: Let's think step by step. " * args.repeat_prompt
prompts = {1: [prompt], 8: [prompt] + [prompt[: 200 + 40 * i] for i in range(7)]}

def timed(model, batch):
    enc = tok(prompts[batch], return_tensors="pt", padding=True).to("cuda")
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=args.steps, min_new_tokens=args.steps, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); total = time.time() - t0
    return {"batch": batch, "prompt_tokens": int(enc["input_ids"].shape[1]), "seconds": total,
            "ms_per_step": 1000 * total / args.steps, "ms_per_step_per_seq": 1000 * total / args.steps / batch}

def profile_step(model, batch=8):
    """Top CUDA kernels for one decode step after a prefill, all hooks on."""
    enc = tok(prompts[batch], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = model(**enc, use_cache=True)
        past = out.past_key_values
        next_ids = out.logits[:, -1:].argmax(-1)
        mask = torch.cat([enc["attention_mask"], torch.ones_like(next_ids)], dim=1)
        model(input_ids=next_ids, attention_mask=mask, past_key_values=past, use_cache=True)  # warm
        from torch.profiler import profile, ProfilerActivity
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(4):
                out = model(input_ids=next_ids, attention_mask=mask, past_key_values=past, use_cache=True)
                next_ids = out.logits[:, -1:].argmax(-1)
                mask = torch.cat([mask, torch.ones_like(next_ids)], dim=1)
            torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60)
    print(table, flush=True)
    return table


report = {}
configs = {"stock": None, "all_hooks": "both", "kv_only": "kv", "act_only": "act"}
for label in args.configs.replace("+", ",").split(","):
    kind = configs[label]
    model, hooks, _ = e22.build_row(args, "bf16" if kind is None else "nar_kmax_asym_g128")
    if hooks is not None and kind != "both":
        hooks.close()
        rot = hooks.rotations
        hooks = e14.RuntimeHooks(model, rot, activation_kind=(None if kind == "kv" else e22.ACTIVATION_KIND), quantize_kv=(kind == "kv"))
        hooks.install()
    try:
        timed(model, 1)  # warm-up
        report[label] = {b: timed(model, b) for b in (1, 8)}
        print(label, json.dumps(report[label]), flush=True)
        if args.profile and kind == "both":
            report[label]["profile"] = profile_step(model)
    finally:
        if hooks is not None: hooks.close()
        del model; torch.cuda.empty_cache()
out = e22.result_dir(args.model) / f"e22_decode_timing_x{args.repeat_prompt}.json"
out.write_text(json.dumps(report, indent=2)); print("written", out)
