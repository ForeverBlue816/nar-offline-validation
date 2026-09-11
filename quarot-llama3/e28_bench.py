#!/usr/bin/env python3
"""E28: end-to-end throughput and memory of QuaRot's integer pipeline with the NAR R4 kernel.

Runs inside the E28 environment (torch 2.4.1, transformers 4.36.2, quarot._CUDA,
fast-hadamard-transform, flash-attn).  Three rows per model, one process per
(model, row):

  select   verify + select the NAR kernel configs for one model (writes
           results/e28/<model>/nar_kernel_selection.json)
  bench    one row: prefill (bs 1, 16; 2048 tokens), decode (bs 1, 128 steps
           after a 2048-token prefill), peak memory -> metrics_<row>.json
  collect  merge the rows into metrics.json, write table.tex and env.txt

Random INT4 weights (QuaRot's benchmark convention), fixed seed, fp16 pipeline
dtype.  Timing: torch.cuda.synchronize around every measurement, 10 warmup
iterations, median and std over 50 runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nar_r4_kernel as nk  # noqa: E402

MODELS = {
    "3b": {"key": "llama32_3b", "hf": "unsloth/Llama-3.2-3B", "label": "Llama-3.2-3B"},
    "8b": {"key": "llama31_8b", "hf": "unsloth/Meta-Llama-3.1-8B", "label": "Llama-3.1-8B"},
}
ROWS = ("fp16", "hadamard", "nar")
ROW_LABELS = {"fp16": "FP16 (bf16 row)", "hadamard": "QuaRot Hadamard", "nar": "PrismQuant k=8"}
RANK = 8
TOLERANCE = {"code_match_fraction_min": 0.999, "relative_l2_max": 2e-3}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def nvidia_smi() -> dict[str, Any]:
    def query(args: list[str]) -> str:
        try:
            return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=30).strip()
        except Exception as error:  # noqa: BLE001
            return f"nvidia-smi failed: {error}"
    return {
        "gpu": query(["--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,"
                      "temperature.gpu,clocks.sm,clocks.mem,pstate", "--format=csv"]),
        "compute_apps": query(["--query-compute-apps=pid,process_name,used_memory", "--format=csv"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def config_path(workdir: Path, hf_id: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(hf_id, "config.json", cache_dir=str(workdir / "cache" / "huggingface" / "hub")))


def require_a40() -> None:
    capability = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if capability != (8, 6):
        raise SystemExit(f"E28 requires sm_86 (A40) for QuaRot's INT4 tensor-core GEMM; got {name} {capability}")
    log(f"device {name} capability {capability}")


# ----------------------------------------------------------------- models ---

def build_model(row: str, config, factors: dict[str, Any] | None, selection: dict[str, Any] | None,
                seed: int) -> torch.nn.Module:
    import modeling_llama3 as ml
    import transformers
    torch.manual_seed(seed)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights():
        if row == "fp16":
            model = ml.QuarotFP16LlamaForCausalLM(config)
        elif row == "hadamard":
            model = ml.QuarotLlamaForCausalLM(config)
        elif row == "nar":
            assert factors is not None and selection is not None
            h128 = nk.sylvester_hadamard_fp16(torch.device("cuda"))
            transforms = [nk.NARDownTransform(layer["y_prime_fp32"], layer["w_h_t_fp32"], RANK, selection, h128)
                          for layer in factors["layers"]]
            model = ml.QuarotNARLlamaForCausalLM(config, transforms)
        else:
            raise ValueError(row)
    torch.set_default_dtype(default)
    ml.install_llama3_rope(model)
    # no_init_weights also skips tie_weights(); Llama-3.2-3B ties lm_head to the
    # input embedding (Llama-3.1-8B does not), and the memory rows must reflect that.
    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()
        assert model.lm_head.weight is model.model.embed_tokens.weight
    # Random weights (QuaRot's benchmark convention).  no_init_weights leaves the
    # fp16 parameters uninitialised; give them finite values so the fp16 pipeline
    # does not run on NaN/Inf.  Linear4bit already holds random INT4 codes; its
    # weight_scales are zero by construction and are set to a small positive value.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dim() == 1:
                parameter.fill_(1.0)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
                                .mul_(0.02).to(parameter.dtype))
        for module in model.modules():
            if module.__class__.__name__ == "Linear4bit":
                module.weight_scales.copy_((torch.rand(module.weight_scales.shape, generator=generator)
                                            * 0.002 + 0.001).to(module.weight_scales.dtype))
    model.logits_last_only = True
    return model.eval().cuda()


def model_bytes(model: torch.nn.Module) -> dict[str, int]:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    nar = sum(m.factor_bytes() for m in model.modules() if isinstance(m, nk.NARDownTransform))
    return {"parameters": params, "buffers": buffers, "total": params + buffers, "nar_factors": nar}


# ---------------------------------------------------------------- timing ---

def summarize(samples: list[float]) -> dict[str, float]:
    return {"median": statistics.median(samples), "mean": statistics.fmean(samples),
            "std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
            "min": min(samples), "max": max(samples), "n": len(samples)}


@torch.no_grad()
def time_prefill(model, batch_size: int, length: int, warmup: int, runs: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(100, 200, (batch_size, length), generator=generator, dtype=torch.int32).cuda()

    def step() -> None:
        model._expected_max_length = length
        out = model(input_ids)
        del out

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    peak = torch.cuda.max_memory_allocated()
    ms = summarize(samples)
    return {"batch_size": batch_size, "prompt_length": length, "ms": ms,
            "tokens_per_s": batch_size * length / (ms["median"] / 1000.0),
            "tokens_per_s_std": batch_size * length * ms["std"] / (ms["median"] ** 2) * 1000.0,
            "peak_memory_bytes": peak, "allocated_before_bytes": baseline, "samples_ms": samples}


@torch.no_grad()
def time_decode(model, batch_size: int, prefill_length: int, steps: int, discard: int,
                warmup: int, runs: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(100, 200, (batch_size, prefill_length), generator=generator, dtype=torch.int32).cuda()
    model._expected_max_length = prefill_length + steps
    out = model(input_ids)
    cache = out.past_key_values
    del out
    gc.collect()
    torch.cuda.empty_cache()
    next_input = torch.full((batch_size, 1), 100, dtype=torch.int32, device="cuda")

    def one_run() -> tuple[float, list[float]]:
        cache.length = prefill_length
        per_step = []
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(next_input, past_key_values=cache)
            torch.cuda.synchronize()
            per_step.append((time.perf_counter() - start) * 1000.0)
        return statistics.fmean(per_step[discard:]), per_step

    for _ in range(warmup):
        one_run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    samples = []
    first_run_steps = None
    for index in range(runs):
        ms_per_token, per_step = one_run()
        samples.append(ms_per_token)
        if index == 0:
            first_run_steps = per_step
    peak = torch.cuda.max_memory_allocated()
    # Diagnostic (not a table column): GPU kernel time per decode step from the
    # profiler, so the CPU-launch-bound part of the wall-clock latency is visible.
    from torch.profiler import ProfilerActivity, profile
    cache.length = prefill_length
    for _ in range(discard):
        model(next_input, past_key_values=cache)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(8):
            model(next_input, past_key_values=cache)
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
    gpu_us = sum(getattr(e, "self_device_time_total", 0.0) for e in events)
    kernel_launches = sum(e.count for e in events)
    return {"batch_size": batch_size, "prefill_length": prefill_length, "steps": steps, "discarded": discard,
            "ms_per_token": summarize(samples), "peak_memory_bytes": peak,
            "allocated_before_bytes": baseline, "samples_ms_per_token": samples,
            "first_run_step_ms": first_run_steps,
            "gpu_kernel_ms_per_token_profiler": gpu_us / 8 / 1000.0,
            "gpu_kernel_launches_per_token_profiler": kernel_launches / 8}


# --------------------------------------------------------------- commands ---

def command_select(args: argparse.Namespace) -> None:
    require_a40()
    spec = MODELS[args.model]
    out_dir = args.results / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    factors = nk.load_factors(args.workdir / "tmp" / "e28" / f"nar_factors_{spec['key']}_k{RANK}.pt",
                              torch.device("cuda"))
    layer0 = factors["layers"][0]
    token_counts = sorted({1, args.prefill_length, args.prefill_length * max(args.batch_sizes)})
    selection, verification = nk.select_configs(layer0["y_prime_fp32"], layer0["w_h_t_fp32"], RANK,
                                                token_counts, args.seed, TOLERANCE)
    # Precision of the selected pipeline on every layer's factors, and of QuaRot's
    # own fp16 Hadamard against its fp32 reference, so the two rows' transform
    # precisions are reported side by side.
    import quarot
    n = factors["n"]
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    sample = (torch.randn((256, n), generator=generator) * 0.5).to("cuda", torch.float16)
    per_layer = []
    for index, layer in enumerate(factors["layers"]):
        module = nk.NARDownTransform(layer["y_prime_fp32"], layer["w_h_t_fp32"], RANK, selection).cuda()
        reference = nk.reference_transform(sample, layer["y_prime_fp32"], layer["w_h_t_fp32"])
        row = nk.precision_row(module(sample), reference)
        row["layer"] = index
        per_layer.append(row)
    hadamard = quarot.nn.OnlineHadamard(n).cuda()
    from quarot.functional.hadamard import matmul_hadU
    had_ref = matmul_hadU(sample.float())
    had_precision = nk.precision_row(hadamard(sample), had_ref)
    payload = {"model": spec["key"], "rank": RANK, "terms": nk.TERMS, "tolerance": TOLERANCE,
               "token_counts": token_counts, "selection": selection,
               "per_layer_precision": per_layer,
               "worst_layer_precision": {key: (min if key == "code_match_fraction" else max)(r[key] for r in per_layer)
                                         for key in ("relative_l2", "max_abs_over_row_absmax", "code_match_fraction")},
               "quarot_hadamard_fp16_precision": had_precision,
               "verification_log": verification, "device": torch.cuda.get_device_name()}
    nk.write_json(out_dir / "nar_kernel_selection.json", payload)
    log(f"selection: {json.dumps({t: (s['proj'], s['tile'], round(s['pipeline_ms'], 5)) for t, s in selection.items()})}")
    log(f"NAR worst-layer precision {payload['worst_layer_precision']}; QuaRot Hadamard fp16 {had_precision}")


def command_bench(args: argparse.Namespace) -> None:
    require_a40()
    import modeling_llama3 as ml
    spec = MODELS[args.model]
    out_dir = args.results / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    smi_before = nvidia_smi()
    log(f"nvidia-smi before run:\n{smi_before['gpu']}\n{smi_before['compute_apps']}")
    config = ml.load_llama3_config(str(config_path(args.workdir, spec["hf"])))
    factors = selection = None
    if args.row == "nar":
        factors = nk.load_factors(args.workdir / "tmp" / "e28" / f"nar_factors_{spec['key']}_k{RANK}.pt",
                                  torch.device("cuda"))
        selection = json.loads((out_dir / "nar_kernel_selection.json").read_text())["selection"]
    model = build_model(args.row, config, factors, selection, args.seed)
    sizes = model_bytes(model)
    log(f"{spec['key']} row={args.row}: model bytes {sizes}")
    result: dict[str, Any] = {"model": spec["key"], "hf_id": spec["hf"], "row": args.row,
                              "row_label": ROW_LABELS[args.row], "seed": args.seed,
                              "weights": "random INT4 codes / random fp16 (QuaRot benchmark convention)",
                              "model_bytes": sizes, "nvidia_smi_before": smi_before,
                              "protocol": {"warmup": args.warmup, "runs": args.runs,
                                           "prefill_length": args.prefill_length,
                                           "decode_steps": args.decode_steps, "decode_discard": args.decode_discard,
                                           "logits": "last position only (all rows)"},
                              "prefill": {}, "decode": None}
    for batch_size in args.batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        row = time_prefill(model, batch_size, args.prefill_length, args.warmup, args.runs, args.seed)
        result["prefill"][f"bs{batch_size}"] = row
        log(f"prefill bs={batch_size}: {row['ms']['median']:.3f} ms (std {row['ms']['std']:.3f}), "
            f"{row['tokens_per_s']:.1f} tok/s, peak {row['peak_memory_bytes'] / 2**30:.3f} GiB")
    gc.collect()
    torch.cuda.empty_cache()
    row = time_decode(model, 1, args.prefill_length, args.decode_steps, args.decode_discard,
                      args.warmup, args.runs, args.seed)
    result["decode"] = row
    log(f"decode bs=1: {row['ms_per_token']['median']:.4f} ms/token (std {row['ms_per_token']['std']:.4f}), "
        f"peak {row['peak_memory_bytes'] / 2**30:.3f} GiB")
    result["nvidia_smi_after"] = nvidia_smi()
    result["hardware"] = {"hostname": platform.node(), "gpu": torch.cuda.get_device_name(),
                          "capability": list(torch.cuda.get_device_capability()),
                          "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    nk.write_json(out_dir / f"metrics_{args.row}.json", result)


def versions() -> dict[str, Any]:
    import flash_attn
    import quarot
    import transformers
    import triton
    import fast_hadamard_transform
    quarot_dir = Path(quarot.__file__).resolve().parents[1]

    def git(*cmd: str, cwd: Path) -> str:
        try:
            return subprocess.check_output(["git", *cmd], cwd=cwd, text=True).strip()
        except Exception as error:  # noqa: BLE001
            return f"unavailable: {error}"
    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__, "transformers": transformers.__version__,
        "flash_attn": flash_attn.__version__,
        "fast_hadamard_transform": getattr(fast_hadamard_transform, "__version__", "1.0.4.post1 (QuaRot submodule)"),
        "cutlass": git("describe", "--tags", "--always", cwd=quarot_dir / "third-party" / "cutlass"),
        "quarot_commit": git("rev-parse", "HEAD", cwd=quarot_dir),
        "quarot_patch": "quarot-llama3/quarot_gqa_decode.patch (GQA in the FlashInfer decode kernel)",
        "nar_repo_commit": git("rev-parse", "HEAD", cwd=HERE.parent),
        "nvcc": subprocess.run(["bash", "-lc", "module load CUDA/12.4.0 2>/dev/null; nvcc --version | tail -1"],
                               capture_output=True, text=True).stdout.strip(),
        "driver": nvidia_smi()["gpu"],
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "hostname": platform.node(),
    }


def command_collect(args: argparse.Namespace) -> None:
    table_rows = []
    for model in args.models:
        out_dir = args.results / model
        rows = {}
        for row in ROWS:
            path = out_dir / f"metrics_{row}.json"
            if path.exists():
                rows[row] = json.loads(path.read_text())
        if not rows:
            continue
        merged: dict[str, Any] = {"model": MODELS[model]["key"], "label": MODELS[model]["label"], "rows": {}}
        base = rows.get("fp16")
        for row, data in rows.items():
            entry = {
                "prefill_tok_s_bs1": data["prefill"]["bs1"]["tokens_per_s"],
                "prefill_tok_s_bs1_std": data["prefill"]["bs1"]["tokens_per_s_std"],
                "prefill_ms_bs1": data["prefill"]["bs1"]["ms"],
                "prefill_tok_s_bs16": data["prefill"]["bs16"]["tokens_per_s"],
                "prefill_tok_s_bs16_std": data["prefill"]["bs16"]["tokens_per_s_std"],
                "prefill_ms_bs16": data["prefill"]["bs16"]["ms"],
                "decode_ms_per_token": data["decode"]["ms_per_token"],
                "decode_gpu_kernel_ms_per_token_profiler": data["decode"].get("gpu_kernel_ms_per_token_profiler"),
                "decode_gpu_kernel_launches_per_token_profiler": data["decode"].get("gpu_kernel_launches_per_token_profiler"),
                "peak_memory_prefill_bs16_bytes": data["prefill"]["bs16"]["peak_memory_bytes"],
                "peak_memory_decode_bytes": data["decode"]["peak_memory_bytes"],
                "model_bytes": data["model_bytes"],
            }
            if base is not None:
                entry["speedup_vs_fp16"] = {
                    "prefill_bs1": entry["prefill_tok_s_bs1"] / base["prefill"]["bs1"]["tokens_per_s"],
                    "prefill_bs16": entry["prefill_tok_s_bs16"] / base["prefill"]["bs16"]["tokens_per_s"],
                    "decode": base["decode"]["ms_per_token"]["median"] / entry["decode_ms_per_token"]["median"],
                }
            merged["rows"][row] = entry
        if "hadamard" in merged["rows"] and "nar" in merged["rows"] and base is not None:
            had = merged["rows"]["hadamard"]["speedup_vs_fp16"]
            nar = merged["rows"]["nar"]["speedup_vs_fp16"]
            merged["percent_of_hadamard_speedup"] = {
                key: 100.0 * (nar[key] - 1.0) / (had[key] - 1.0) if had[key] != 1.0 else float("nan") for key in had}
            merged["nar_minus_hadamard"] = {
                "decode_ms_per_token": merged["rows"]["nar"]["decode_ms_per_token"]["median"]
                - merged["rows"]["hadamard"]["decode_ms_per_token"]["median"],
                "peak_memory_decode_bytes": merged["rows"]["nar"]["peak_memory_decode_bytes"]
                - merged["rows"]["hadamard"]["peak_memory_decode_bytes"],
                "peak_memory_prefill_bs16_bytes": merged["rows"]["nar"]["peak_memory_prefill_bs16_bytes"]
                - merged["rows"]["hadamard"]["peak_memory_prefill_bs16_bytes"],
                "model_bytes": merged["rows"]["nar"]["model_bytes"]["total"]
                - merged["rows"]["hadamard"]["model_bytes"]["total"],
            }
        selection_path = out_dir / "nar_kernel_selection.json"
        if selection_path.exists():
            sel = json.loads(selection_path.read_text())
            merged["nar_kernel"] = {"selection": sel["selection"], "worst_layer_precision": sel["worst_layer_precision"],
                                    "quarot_hadamard_fp16_precision": sel["quarot_hadamard_fp16_precision"]}
        merged["raw"] = rows
        nk.write_json(out_dir / "metrics.json", merged)
        table_rows.append(merged)
    env = versions()
    (args.results / "env.txt").write_text("\n".join(f"{key}: {value}" for key, value in env.items()) + "\n")
    (args.results / "table.tex").write_text(latex_table(table_rows))
    log(f"wrote {args.results / 'table.tex'} and env.txt")


def latex_table(models: list[dict[str, Any]]) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{End-to-end deployment on one NVIDIA A40 (sm\_86), QuaRot's INT4 pipeline "
             r"(CUTLASS INT4 GEMM, INT4 KV cache), random INT4 weights, prompt length 2048. "
             r"Rows 2 and 3 differ only in the online transform at the down-projection input. "
             r"Speedup is decode latency relative to the FP16 row; \%~of Hadamard speedup is "
             r"$(s_{\mathrm{PrismQuant}}-1)/(s_{\mathrm{Hadamard}}-1)$ for decode. Peak memory is during decode.}",
             r"\label{tab:e28}",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Model & Method & Prefill tok/s (bs1) & Prefill tok/s (bs16) & Decode ms/tok & "
             r"Peak mem (GB) & Speedup vs FP16 & \% of Hadamard speedup \\",
             r"\midrule"]
    for model in models:
        first = True
        for row in ROWS:
            if row not in model["rows"]:
                continue
            entry = model["rows"][row]
            speed = entry.get("speedup_vs_fp16", {}).get("decode")
            pct = model.get("percent_of_hadamard_speedup", {}).get("decode") if row == "nar" else None
            label = {"fp16": "FP16", "hadamard": "Hadamard (QuaRot)", "nar": "PrismQuant $k{=}8$"}[row]
            lines.append(" & ".join([
                model["label"] if first else "",
                label,
                f"{entry['prefill_tok_s_bs1']:,.0f}",
                f"{entry['prefill_tok_s_bs16']:,.0f}",
                f"{entry['decode_ms_per_token']['median']:.2f} $\\pm$ {entry['decode_ms_per_token']['std']:.2f}",
                f"{entry['peak_memory_decode_bytes'] / 1e9:.2f}",
                "1.00" if row == "fp16" else (f"{speed:.2f}" if speed else "--"),
                f"{pct:.0f}\\%" if pct is not None else ("100\\%" if row == "hadamard" else "--"),
            ]) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "bench", "collect"))
    parser.add_argument("--workdir", type=Path, default=Path(os.environ.get("NAR_WORKDIR", "/projects/_hdd/nar/nar-validation")))
    parser.add_argument("--results", type=Path, default=HERE.parent / "results" / "e28")
    parser.add_argument("--model", choices=tuple(MODELS), default="3b")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--row", choices=ROWS, default="hadamard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill-length", type=int, default=2048)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--decode-discard", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    {"select": command_select, "bench": command_bench, "collect": command_collect}[args.command](args)


if __name__ == "__main__":
    main()
