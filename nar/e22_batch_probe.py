"""Why does batched greedy decoding disagree with batch-1 under the quantizer hooks?

The 8B gate scored 83.0 at batch 1 twice and 51.0 at batch 8. This runs one
padded 2-sequence batch and the same two sequences alone, with and without
the hooks, on a small family member, records what mask the attention hook
receives at prefill and at the first decode step, and compares the tokens.
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nar import e22_qwen3_family as e22, e19_qwen3_e2e as e19, e14_w4a4kv4 as e14, experiment as base

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", required=True)
parser.add_argument("--model", default="qwen3_0.6b_base")
parser.add_argument("--row", default="nar_kmax_asym_g128")
parser.add_argument("--new-tokens", type=int, default=24)
args = parser.parse_args()
e22.WORKDIR = Path(args.workdir).resolve()
args.seed, args.weight_row_batch, args.round_trip_tolerance, args.seq_len = 0, 256, 1e-6, 2048
e22.configure(args.model)
base.setup_logging(e22.WORKDIR, f"e22-batch-probe-{args.model}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(e19.MODEL_ID, cache_dir=str(e22.WORKDIR / "cache" / "huggingface"))
tok.padding_side = "left"
prompts = ["Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: Let's think step by step.",
           "Q: What is 7 times 8?\nA:"]
seen = []

def wrap(hooks):
    inner = hooks.attention
    def logged(module, query, key, value, attention_mask, **kw):
        if len(seen) < 4:
            seen.append({"q_len": int(query.shape[-2]), "kv_len": int(key.shape[-2]), "batch": int(key.shape[0]),
                         "mask": None if attention_mask is None else
                         {"dtype": str(attention_mask.dtype), "shape": list(attention_mask.shape),
                          "min": float(attention_mask.float().min()), "max": float(attention_mask.float().max())}})
        return inner(module, query, key, value, attention_mask, **kw)
    hooks.attention = logged

def generate(model, texts):
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=args.new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

report = {"model": args.model, "row": args.row}
for label in ("stock", args.row):
    row = "bf16" if label == "stock" else args.row
    model, hooks, _ = e22.build_row(args, row)
    if hooks is not None:
        hooks.close(); wrap(hooks); hooks.install()
    try:
        seen.clear()
        single = [generate(model, [p])[0] for p in prompts]
        seen_single = list(seen); seen.clear()
        batched = generate(model, prompts)
        report[label] = {"single": single, "batched": batched, "agree": [a == b for a, b in zip(single, batched)],
                         "masks_single": seen_single, "masks_batched": list(seen)}
        print(label, json.dumps(report[label], indent=1)[:3000], flush=True)
    finally:
        if hooks is not None: hooks.close()
        del model; torch.cuda.empty_cache()
out = e22.result_dir(args.model) / "e22_batch_probe.json"
out.write_text(json.dumps(report, indent=2)); print("written", out)
