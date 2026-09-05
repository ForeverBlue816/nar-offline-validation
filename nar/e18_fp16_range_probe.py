"""Does the fp16 metadata of the activation quantizer survive at 70B scale?

The E18 Llama-3.1-70B Hadamard row reads PPL 1.5e4 against bf16 3.105, and the
exact-transpose rerun reproduces it, so the bf16 weight fold is not the cause and
the rotation-only control passes.  That leaves the quantizer itself.

``dynamic_asym_int4`` stores one fp16 scale and one fp16 real-valued zero per
group.  Two things can go wrong in fp16 and neither is guarded:

  * ``zero16 = lo.to(float16)`` overflows to +-inf once |lo| > 65504.
  * ``scale16`` underflows to exactly 0 for a raw scale below the fp16
    subnormal floor; the guard only rejects a raw scale that is exactly 0 in
    fp32, so the subsequent division by the fp16 zero produces inf or nan.

Either makes the dequantized activation non-finite and would destroy the model
while leaving every orthogonality and round-trip check clean.  This probe counts
both events at both quantization sites, per rotation, without evaluating
perplexity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nar import activation_experiments as act  # noqa: E402
from nar import e18_70b as e18  # noqa: E402
from nar import e18_v2  # noqa: E402
from nar import experiment as base  # noqa: E402

LOG = base.LOG
FP16_MAX = 65504.0
FP16_MIN_SUBNORMAL = 5.960464477539063e-08
FP16_MIN_NORMAL = 6.103515625e-05


def hazards(x: torch.Tensor, group_size: int) -> dict[str, float]:
    xg = base.group_view(x.float(), group_size)
    lo = xg.amin(dim=-1, keepdim=True)
    hi = xg.amax(dim=-1, keepdim=True)
    raw_scale = (hi - lo) / base.QMAX
    scale16 = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale)).to(torch.float16)
    zero16 = lo.to(torch.float16)
    deq = torch.round((xg - zero16.float()) / scale16.float()).clamp_(0, base.QMAX)
    deq = deq * scale16.float() + zero16.float()
    groups = raw_scale.numel()
    return {
        "groups": groups,
        "abs_max": float(x.abs().max()),
        "zero_overflow_groups": int((~torch.isfinite(zero16)).sum()),
        "scale_overflow_groups": int((~torch.isfinite(scale16)).sum()),
        "scale_underflow_to_zero_groups": int(((raw_scale > 0) & (scale16.float() == 0)).sum()),
        "scale_subnormal_groups": int(((raw_scale > 0) & (raw_scale < FP16_MIN_NORMAL)).sum()),
        "nonfinite_dequantized": int((~torch.isfinite(deq)).sum()),
        "values": deq.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--model-key", default="llama31_70b", choices=tuple(e18_v2.MODELS))
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--methods", nargs="+", default=["hadamard", "nar_k8", "nar_kmax"])
    parser.add_argument("--max-layers", type=int, default=0)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, f"e18-fp16-probe-{args.model_key}")
    base.seed_everything(args.seed)
    model_id = e18_v2.MODELS[args.model_key]
    model = e18.load_sharded_model(model_id, workdir)
    layers = e18.selected_layers(model, args.max_layers)
    dimensions = {"qkv": int(model.config.hidden_size),
                  "down": int(model.config.intermediate_size)}
    root = e18_v2.factor_root(workdir, args.model_key, "e18")
    tokens = base.prepare_token_chunks(model_id, "test", 0, args.sequences, args.seq_len, workdir)

    rows: list[dict] = []
    for method in ["identity"] + list(args.methods):
        rotations = None
        if method != "identity":
            rotations = e18_v2.ShardedRotationsAt(args, model, root, method, dimensions, layers)
        worst: dict[str, dict] = {}
        handles = []

        def make(site: str, layer: int):
            def hook(_module, inputs):
                x = inputs[0].detach()
                if rotations is not None:
                    x = rotations.apply(site, layer, x.float())
                stat = hazards(x, args.group_size)
                stat.update({"site": site, "layer": layer})
                key = site
                if key not in worst or stat["abs_max"] > worst[key]["abs_max"]:
                    worst[key] = stat
                acc = worst.setdefault(f"{key}_totals", {"zero_overflow_groups": 0,
                                                         "scale_underflow_to_zero_groups": 0,
                                                         "nonfinite_dequantized": 0, "groups": 0})
                for field in acc:
                    acc[field] += stat[field]
            return hook

        for layer, block in enumerate(model.model.layers[:layers]):
            handles.append(block.self_attn.q_proj.register_forward_pre_hook(make("qkv", layer)))
            handles.append(block.mlp.down_proj.register_forward_pre_hook(make("down", layer)))
        with torch.inference_mode():
            for index in range(tokens.shape[0]):
                model(input_ids=tokens[index:index + 1].to(model.device), use_cache=False)
        for handle in handles:
            handle.remove()
        for site in ("qkv", "down"):
            entry = {"model": args.model_key, "method": method, "site": site,
                     **{k: v for k, v in worst[site].items() if k != "site"},
                     **{f"total_{k}": v for k, v in worst[f"{site}_totals"].items()}}
            rows.append(entry)
            LOG.info("%s %s %s: abs_max=%.4g zero_overflow=%d underflow=%d nonfinite=%d/%d",
                     method, site, f"worst layer {entry['layer']}", entry["abs_max"],
                     entry["total_zero_overflow_groups"],
                     entry["total_scale_underflow_to_zero_groups"],
                     entry["total_nonfinite_dequantized"], entry["total_groups"])
        del rotations
        torch.cuda.empty_cache()

    out = workdir / "results" / args.model_key / "e18_fp16_range_probe.csv"
    base.write_csv(out, rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
