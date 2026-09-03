#!/usr/bin/env python3
"""E17 fused Triton NAR-R4 + asymmetric INT4 benchmark.

One Triton launch per token row performs the compact rank-8 WY update, frozen
permutation/signs, block-H128, and group-128 dynamic asymmetric INT4 packing.
The matched baseline fuses signs, block-H128, and the identical quantizer.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from . import activation_experiments as act
    from . import experiment as base
    from .e12_wy import compact_wy
except ImportError:
    import activation_experiments as act
    import experiment as base
    from e12_wy import compact_wy


N = 8192
HIDDEN = 3072
GROUP = 128
GROUPS = N // GROUP
K = 8


@triton.jit
def _fwht_stage(value, width: tl.constexpr):
    blocks = tl.reshape(value, (64, 128 // (2 * width), 2, width))
    split_last = tl.permute(blocks, 0, 1, 3, 2)
    left, right = tl.split(split_last)
    joined = tl.join(left + right, left - right)
    contiguous_halves = tl.permute(joined, 0, 1, 3, 2)
    return tl.reshape(contiguous_halves, (64, 128))


@triton.jit
def _fwht128(value):
    value = _fwht_stage(value, 1)
    value = _fwht_stage(value, 2)
    value = _fwht_stage(value, 4)
    value = _fwht_stage(value, 8)
    value = _fwht_stage(value, 16)
    value = _fwht_stage(value, 32)
    value = _fwht_stage(value, 64)
    return value * 0.08838834764831845


@triton.jit
def _quantize_pack_store(value, codes_ptr, scales_ptr, zeros_ptr, row):
    lo = tl.min(value, axis=1)
    hi = tl.max(value, axis=1)
    raw_scale = (hi - lo) * (1.0 / 15.0)
    scale = tl.where(raw_scale > 0.0, raw_scale, 1.0).to(tl.float16)
    zero = lo.to(tl.float16)
    scale_f = tl.reshape(scale.to(tl.float32), (64, 1))
    zero_f = tl.reshape(zero.to(tl.float32), (64, 1))
    quant = tl.floor((value - zero_f) / scale_f + 0.5)
    quant = tl.maximum(0.0, tl.minimum(15.0, quant)).to(tl.uint8)
    pairs = tl.reshape(quant, (64, 64, 2))
    low, high = tl.split(pairs)
    packed = low | (high << 4)
    code_offsets = tl.arange(0, 4096)
    group_offsets = tl.arange(0, 64)
    tl.store(codes_ptr + row * 4096 + code_offsets, tl.reshape(packed, (4096,)))
    tl.store(scales_ptr + row * 64 + group_offsets, scale)
    tl.store(zeros_ptr + row * 64 + group_offsets, zero)


@triton.jit
def fused_nar_int4_kernel(x_ptr, w_ptr, y_ptr, source_for_target_ptr, signs_ptr,
                          codes_ptr, scales_ptr, zeros_ptr):
    row = tl.program_id(0)
    offsets = tl.arange(0, 8192)
    x = tl.load(x_ptr + row * 8192 + offsets).to(tl.float32)
    p0 = tl.sum(x * tl.load(w_ptr + offsets * 8 + 0), axis=0)
    p1 = tl.sum(x * tl.load(w_ptr + offsets * 8 + 1), axis=0)
    p2 = tl.sum(x * tl.load(w_ptr + offsets * 8 + 2), axis=0)
    p3 = tl.sum(x * tl.load(w_ptr + offsets * 8 + 3), axis=0)
    p4 = tl.sum(x * tl.load(w_ptr + offsets * 8 + 4), axis=0)
    p5 = tl.sum(x * tl.load(w_ptr + offsets * 8 + 5), axis=0)
    p6 = tl.sum(x * tl.load(w_ptr + offsets * 8 + 6), axis=0)
    p7 = tl.sum(x * tl.load(w_ptr + offsets * 8 + 7), axis=0)
    source = tl.load(source_for_target_ptr + offsets).to(tl.int32)
    # Register gather makes this a literal one-read pass over x: x is loaded
    # once above for WY, then reordered without another global-memory x load.
    permuted = tl.gather(x, source, axis=0)
    permuted -= p0 * tl.load(y_ptr + source * 8 + 0)
    permuted -= p1 * tl.load(y_ptr + source * 8 + 1)
    permuted -= p2 * tl.load(y_ptr + source * 8 + 2)
    permuted -= p3 * tl.load(y_ptr + source * 8 + 3)
    permuted -= p4 * tl.load(y_ptr + source * 8 + 4)
    permuted -= p5 * tl.load(y_ptr + source * 8 + 5)
    permuted -= p6 * tl.load(y_ptr + source * 8 + 6)
    permuted -= p7 * tl.load(y_ptr + source * 8 + 7)
    permuted *= tl.load(signs_ptr + offsets)
    transformed = _fwht128(tl.reshape(permuted, (64, 128)))
    _quantize_pack_store(transformed, codes_ptr, scales_ptr, zeros_ptr, row)


@triton.jit
def fused_hadamard_int4_kernel(x_ptr, signs_ptr, codes_ptr, scales_ptr, zeros_ptr):
    row = tl.program_id(0)
    offsets = tl.arange(0, 8192)
    value = tl.load(x_ptr + row * 8192 + offsets).to(tl.float32)
    value *= tl.load(signs_ptr + offsets)
    transformed = _fwht128(tl.reshape(value, (64, 128)))
    _quantize_pack_store(transformed, codes_ptr, scales_ptr, zeros_ptr, row)


def factor_path(workdir: Path) -> Path:
    return (
        workdir / "activations" / "llama32_3b" / "e11_calibration" / "factors"
        / "nar_b128_k8" / "down_layer_00.pt"
    )


def signs_for(device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(act._seed(seed, 0, 0, "down"))
    return torch.randint(0, 2, (N,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1).to(device)


def source_for_target(factor: act.RotationFactor) -> torch.Tensor:
    result = torch.empty_like(factor.source_order)
    result[factor.target_order] = factor.source_order
    return result.to(torch.int32)


def allocate_outputs(tokens: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((tokens, N // 2), dtype=torch.uint8, device=device),
        torch.empty((tokens, GROUPS), dtype=torch.float16, device=device),
        torch.empty((tokens, GROUPS), dtype=torch.float16, device=device),
    )


def launch_nar(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, permutation: torch.Tensor,
               signs: torch.Tensor, outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    fused_nar_int4_kernel[(x.shape[0],)](
        x, w, y, permutation, signs, *outputs, num_warps=8, num_stages=1,
    )


def launch_hadamard(x: torch.Tensor, signs: torch.Tensor,
                    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    fused_hadamard_int4_kernel[(x.shape[0],)](
        x, signs, *outputs, num_warps=8, num_stages=1,
    )


def unpack(codes: torch.Tensor) -> torch.Tensor:
    low = codes & 0x0F
    high = codes >> 4
    return torch.stack((low, high), dim=-1).reshape(codes.shape[0], N)


def dequantize(outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    codes, scales, zeros = outputs
    q = unpack(codes).reshape(codes.shape[0], GROUPS, GROUP).float()
    return (q * scales.float().unsqueeze(-1) + zeros.float().unsqueeze(-1)).reshape(codes.shape[0], N)


def reference_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dequant, scales, zeros, codes = base.dynamic_asym_int4(value, GROUP)
    pairs = codes.reshape(codes.shape[0], GROUPS, GROUP // 2, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)
    return dequant.float(), packed.reshape(codes.shape[0], N // 2), scales, zeros


def benchmark(fn: Callable[[], Any], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(stop) / repeats)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E17 requires CUDA")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e17-fused-r4")
    result_dir = workdir / "results" / "llama32_3b"
    done = result_dir / "E17_DONE.json"
    if done.exists():
        return
    device = torch.device("cuda")
    factor = act.RotationFactor.load(factor_path(workdir), device)
    w, y = compact_wy(factor.reflectors, factor.active)
    permutation = source_for_target(factor)
    signs = signs_for(device, args.seed)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    probe = torch.randn((args.verify_tokens, N), generator=generator, dtype=torch.float32).to(device, torch.bfloat16)
    nar_outputs = allocate_outputs(args.verify_tokens, device)
    had_outputs = allocate_outputs(args.verify_tokens, device)
    launch_nar(probe, w, y, permutation, signs, nar_outputs)
    launch_hadamard(probe, signs, had_outputs)
    torch.cuda.synchronize()
    nar_reference_value = factor.apply(probe.float(), signs)
    had_reference_value = act.ext._fast_walsh_hadamard(
        (probe.float() * signs).reshape(-1, GROUPS, GROUP)
    ).reshape_as(probe)
    nar_ref_deq, nar_ref_codes, nar_ref_scales, nar_ref_zeros = reference_quant(nar_reference_value)
    had_ref_deq, had_ref_codes, had_ref_scales, had_ref_zeros = reference_quant(had_reference_value)
    nar_deq = dequantize(nar_outputs)
    had_deq = dequantize(had_outputs)
    verification = {
        "verify_tokens": args.verify_tokens,
        "nar_code_match_fraction": float((nar_outputs[0] == nar_ref_codes).float().mean()),
        "nar_scale_max_abs": float((nar_outputs[1] - nar_ref_scales).abs().max()),
        "nar_zero_max_abs": float((nar_outputs[2] - nar_ref_zeros).abs().max()),
        "nar_dequant_max_abs": float((nar_deq - nar_ref_deq).abs().max()),
        "nar_dequant_relative_l2": float((nar_deq - nar_ref_deq).norm() / nar_ref_deq.norm()),
        "hadamard_code_match_fraction": float((had_outputs[0] == had_ref_codes).float().mean()),
        "hadamard_scale_max_abs": float((had_outputs[1] - had_ref_scales).abs().max()),
        "hadamard_zero_max_abs": float((had_outputs[2] - had_ref_zeros).abs().max()),
        "hadamard_dequant_max_abs": float((had_deq - had_ref_deq).abs().max()),
        "hadamard_dequant_relative_l2": float((had_deq - had_ref_deq).norm() / had_ref_deq.norm()),
        "bf16_tolerance": args.bf16_tolerance,
    }
    if verification["nar_dequant_max_abs"] > args.bf16_tolerance:
        raise AssertionError(json.dumps(verification, indent=2))
    if verification["hadamard_dequant_max_abs"] > args.bf16_tolerance:
        raise AssertionError(json.dumps(verification, indent=2))
    del probe, nar_outputs, had_outputs, nar_reference_value, had_reference_value
    del nar_ref_deq, nar_ref_codes, nar_ref_scales, nar_ref_zeros, had_ref_deq, had_ref_codes
    del had_ref_scales, had_ref_zeros, nar_deq, had_deq
    gc.collect()
    torch.cuda.empty_cache()

    weight = torch.randn((HIDDEN, N), generator=generator, dtype=torch.float32).to(device, torch.bfloat16)
    timings: list[dict[str, Any]] = []
    repeat_map = {1: args.repeats_small, 32: args.repeats_small, 2048: args.repeats_large}
    for tokens in args.tokens:
        x = torch.randn((tokens, N), generator=generator, dtype=torch.float32).to(device, torch.bfloat16)
        nar_outputs = allocate_outputs(tokens, device)
        had_outputs = allocate_outputs(tokens, device)
        repeats = repeat_map.get(tokens, args.repeats_large)
        nar_ms = benchmark(lambda: launch_nar(x, w, y, permutation, signs, nar_outputs), args.warmup, repeats)
        had_ms = benchmark(lambda: launch_hadamard(x, signs, had_outputs), args.warmup, repeats)
        matmul_ms = benchmark(lambda: F.linear(x, weight), args.warmup, repeats)
        timings.append({
            "tokens": tokens, "dtype": "bf16", "k": K, "n": N, "hidden": HIDDEN,
            "nar_fused_ms": nar_ms, "hadamard_fused_ms": had_ms, "down_matmul_ms": matmul_ms,
            "nar_overhead_vs_hadamard_ratio": nar_ms / had_ms,
            "nar_ratio_vs_down_matmul": nar_ms / matmul_ms,
            "hadamard_ratio_vs_down_matmul": had_ms / matmul_ms,
            "nar_under_10pct_down_matmul": nar_ms / matmul_ms < 0.10,
            "transform_flops_nar": 4 * N * K + N * int(math.log2(GROUP)),
            "transform_flops_hadamard": N * int(math.log2(GROUP)),
            "down_matmul_flops": 2 * N * HIDDEN,
            "nar_transform_flop_ratio_vs_matmul": (4 * N * K + N * int(math.log2(GROUP))) / (2 * N * HIDDEN),
            "warmup": args.warmup, "repeats": repeats,
        })
        del x, nar_outputs, had_outputs
    base.write_csv(result_dir / "e17_fused_r4_timings.csv", timings)
    at_2048 = next(row for row in timings if row["tokens"] == 2048)
    base.atomic_json(done, {
        "model": "llama32_3b", "site": "down_proj input", "layer": 0,
        "kernel": "single Triton launch and one global x load: compact rank-8 WY, register-gathered frozen permutation, signs, block-H128, dynamic asymmetric group-128 INT4 pack",
        "matched_baseline": "single Triton launch: frozen signs, block-H128, identical dynamic asymmetric group-128 INT4 pack",
        "outputs": "packed uint8 (two INT4 codes/byte), fp16 scale, fp16 real-valued zero per group",
        "input_dtype": "bf16", "verification": verification, "timings": timings,
        "under_10pct_at_2048": at_2048["nar_under_10pct_down_matmul"],
        "e12_superseded": at_2048["nar_under_10pct_down_matmul"],
        "one_token_note": "launch-bound; the same launch-bound regime applies to the matched unfused/fused Hadamard baseline",
        "hardware": base.hardware_info(),
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--verify-tokens", type=int, default=4)
    result.add_argument("--bf16-tolerance", type=float, default=0.03125)
    result.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 2048])
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--repeats-small", type=int, default=100)
    result.add_argument("--repeats-large", type=int, default=30)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
