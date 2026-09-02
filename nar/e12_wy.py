#!/usr/bin/env python3
"""E12 compact-WY deployment benchmark for the down-projection NAR R4."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
except ImportError:
    import activation_experiments as act
    import experiment as base


LOG = logging.getLogger("nar")
MODEL = "llama32_3b"
N = 8192
HIDDEN = 3072
BLOCK = 128
RANKS = (16, 32, 64)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _factor_path(workdir: Path, rank: int) -> Path:
    if rank == 64:
        return workdir / "activations" / MODEL / "activation_factors" / "down_layer_00.pt"
    return (
        workdir / "activations" / MODEL / "e11_calibration" / "factors"
        / f"nar_b128_k{rank}" / "down_layer_00.pt"
    )


def compact_wy(reflectors: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return W,Y such that prod_i(I-2v_i v_i^T) = I-WY^T."""
    columns_w: list[torch.Tensor] = []
    columns_y: list[torch.Tensor] = []
    for index in range(reflectors.shape[0]):
        if not bool(active[index]):
            continue
        vector = reflectors[index].float()
        if columns_w:
            w = torch.stack(columns_w, dim=1)
            y = torch.stack(columns_y, dim=1)
            new_w = 2.0 * (vector - w @ (y.T @ vector))
        else:
            new_w = 2.0 * vector
        columns_w.append(new_w)
        columns_y.append(vector)
    if not columns_w:
        n = reflectors.shape[1]
        empty = reflectors.new_empty((n, 0), dtype=torch.float32)
        return empty, empty
    return torch.stack(columns_w, dim=1), torch.stack(columns_y, dim=1)


@dataclass
class WYFactor:
    factor: act.RotationFactor
    w: torch.Tensor
    y: torch.Tensor

    def apply_g(self, value: torch.Tensor) -> torch.Tensor:
        rows = value.float().reshape(-1, self.factor.n)
        projected = rows @ self.w
        return torch.addmm(rows, projected, self.y.T, beta=1.0, alpha=-1.0).reshape_as(value)

    def apply(self, value: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        rows = self.apply_g(value).reshape(-1, self.factor.n)
        permuted = torch.empty_like(rows)
        permuted[:, self.factor.target_order] = rows[:, self.factor.source_order]
        blocks = (permuted * signs).reshape(-1, self.factor.n // self.factor.b, self.factor.b)
        return act.ext._fast_walsh_hadamard(blocks).reshape(shape)


def _benchmark(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
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
        raise RuntimeError("E12 requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e12-compact-wy")
    result_dir = workdir / "results" / MODEL
    done = result_dir / "E12_DONE.json"
    if done.exists():
        LOG.info("E12 result exists: %s", done)
        return
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    verification: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    down_flops = 2 * N * HIDDEN
    had_flops = N * int(math.log2(N)) + N
    weight = (torch.randn((HIDDEN, N), generator=generator, dtype=torch.float32) / math.sqrt(N)).to(
        device=device, dtype=torch.bfloat16
    )
    signs = torch.randint(0, 2, (N,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1).to(device)
    for rank in RANKS:
        factor = act.RotationFactor.load(_factor_path(workdir, rank), device)
        reflectors = int(factor.active.sum())
        if reflectors != rank:
            raise RuntimeError(f"rank {rank} factor has {reflectors} active reflectors")
        w, y = compact_wy(factor.reflectors, factor.active)
        wy = WYFactor(factor, w, y)

        probe = torch.randn((args.verify_rows, N), generator=generator).to(device)
        sequential_g = act.apply_reflectors(probe, factor.reflectors, factor.active)
        wy_g = wy.apply_g(probe)
        g_diff = wy_g - sequential_g
        dense_chunks: list[torch.Tensor] = []
        eye = torch.eye(N, device=device, dtype=torch.float32)
        for start in range(0, N, args.dense_row_batch):
            dense_chunks.append(factor.apply(eye[start : start + args.dense_row_batch], signs))
        dense_rt = torch.cat(dense_chunks, 0)
        reference = probe @ dense_rt
        observed = wy.apply(probe, signs)
        difference = observed - reference
        verification.append({
            "model": MODEL, "layer": 0, "k": rank,
            "wy_vs_sequential_g_max_abs_error": float(g_diff.abs().max()),
            "wy_vs_sequential_g_relative_l2_error": float(g_diff.norm() / sequential_g.norm()),
            "wy_full_vs_dense_max_abs_error": float(difference.abs().max()),
            "wy_full_vs_dense_relative_l2_error": float(difference.norm() / reference.norm()),
            "verify_rows": args.verify_rows,
        })
        if float(difference.abs().max()) > args.fp32_tolerance:
            raise AssertionError(f"k={rank} compact WY failed fp32 tolerance: {float(difference.abs().max())}")
        del probe, sequential_g, wy_g, g_diff, eye, dense_chunks, dense_rt, reference, observed, difference
        torch.cuda.empty_cache()

        wy_flops = 4 * N * rank
        r4_flops = wy_flops + N * int(math.log2(BLOCK)) + N
        for tokens in args.benchmark_tokens:
            x = torch.randn((tokens, N), generator=generator).to(device)
            x_bf16 = x.to(torch.bfloat16)
            wy_g_ms = _benchmark(lambda: wy.apply_g(x), args.warmup, args.repeats)
            r4_ms = _benchmark(lambda: wy.apply(x, signs), args.warmup, args.repeats)
            had_ms = _benchmark(lambda: act.full_hadamard_rows(x, signs), args.warmup, args.repeats)
            matmul_ms = _benchmark(
                lambda: torch.nn.functional.linear(x_bf16, weight), args.warmup, args.repeats
            )
            timings.append({
                "model": MODEL, "layer": 0, "tokens": tokens, "n": N, "hidden": HIDDEN,
                "k": rank, "wy_columns": rank, "wy_g_flops_per_token": wy_flops,
                "r4_flops_per_token": r4_flops, "unfused_hadamard_flops_per_token": had_flops,
                "down_matmul_flops_per_token": down_flops,
                "wy_g_flop_ratio_vs_matmul": wy_flops / down_flops,
                "r4_flop_ratio_vs_matmul": r4_flops / down_flops,
                "wy_g_ms": wy_g_ms, "r4_wy_ms": r4_ms, "unfused_hadamard_ms": had_ms,
                "down_matmul_ms": matmul_ms,
                "r4_wall_ratio_vs_unfused_hadamard": r4_ms / had_ms,
                "r4_wall_ratio_vs_down_matmul": r4_ms / matmul_ms,
                "unfused_hadamard_wall_ratio_vs_down_matmul": had_ms / matmul_ms,
                "r4_under_10pct_matmul_wall": r4_ms / matmul_ms < 0.10,
            })
            del x, x_bf16
        LOG.info("E12 k=%d complete", rank)
        del factor, w, y, wy
        gc.collect()
        torch.cuda.empty_cache()
    base.write_csv(result_dir / "e12_wy_verification.csv", verification)
    base.write_csv(result_dir / "e12_wy_online_cost.csv", timings)
    at_2048 = [row for row in timings if row["tokens"] == 2048]
    base.atomic_json(done, {
        "model": MODEL, "site": "down_proj input", "layer": 0,
        "ranks": list(RANKS), "tokens": list(args.benchmark_tokens),
        "compact_wy": "G=I-WY^T with W,Y in R^(8192 x k); online G uses two GEMM kernel launches; subtraction is fused into the addmm epilogue",
        "full_r4": "compact WY G, fixed permutation/sign, then block H128",
        "benchmark_dtype": "rotation fp32; down_proj matmul bf16, matching E6",
        "unfused_hadamard": "PyTorch staged FWHT with no fused custom kernel",
        "verification": verification,
        "fp32_tolerance": args.fp32_tolerance,
        "under_10pct_at_2048": {str(row["k"]): row["r4_under_10pct_matmul_wall"] for row in at_2048},
        "one_token_interpretation": "kernel-launch-bound for both compact WY R4 and unfused Hadamard",
        "benchmark": {"warmup": args.warmup, "repeats": args.repeats},
        "hardware": base.hardware_info(),
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--verify-rows", type=int, default=8)
    parser.add_argument("--dense-row-batch", type=int, default=256)
    parser.add_argument("--benchmark-tokens", type=int, nargs="+", default=[1, 32, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--fp32-tolerance", type=float, default=1e-5)
    return parser


if __name__ == "__main__":
    run(parser().parse_args())
