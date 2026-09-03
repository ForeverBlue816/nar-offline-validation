#!/usr/bin/env python3
"""Two-launch R4 v2 kernels with the signed permutation folded offline.

Kernel A computes the rank-k projection u = x_perm @ Y' as a streaming
reduction that is split over the channel axis, so the launch keeps every SM
busy instead of serialising one token row per program.  Kernel B owns one or
more token/group-128 tiles, folds the split partials back into u, and applies
the block Hadamard, the rank-k correction, and the group-128 INT4 quantizer.
Neither kernel ever holds a complete 8K/14K activation row in registers.

The matched baseline is Kernel B with the two correction lines removed; it
shares the quantize/pack path, the grid, and the autotune space so that the
NAR/Hadamard wall-clock ratio isolates the marginal cost of the rank-k update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl


GROUP = 128
# Kernel A aims for at least this many concurrent programs before it stops
# splitting the channel axis; below it the launch cannot fill a modern GPU.
MIN_PROJECTION_PROGRAMS = 512
MAX_SPLITS = 32


@triton.jit
def _fwht_stage(value, BLOCK_T: tl.constexpr, width: tl.constexpr):
    blocks = tl.reshape(value, (BLOCK_T, 128 // (2 * width), 2, width))
    split_last = tl.permute(blocks, 0, 1, 3, 2)
    left, right = tl.split(split_last)
    joined = tl.join(left + right, left - right)
    contiguous_halves = tl.permute(joined, 0, 1, 3, 2)
    return tl.reshape(contiguous_halves, (BLOCK_T, 128))


@triton.jit
def _fwht128(value, BLOCK_T: tl.constexpr):
    value = _fwht_stage(value, BLOCK_T, 1)
    value = _fwht_stage(value, BLOCK_T, 2)
    value = _fwht_stage(value, BLOCK_T, 4)
    value = _fwht_stage(value, BLOCK_T, 8)
    value = _fwht_stage(value, BLOCK_T, 16)
    value = _fwht_stage(value, BLOCK_T, 32)
    value = _fwht_stage(value, BLOCK_T, 64)
    return value * 0.08838834764831845


@triton.jit
def _rank_correction(value, u, w_h_ptr, group, channels,
                     N: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr):
    """Subtract the rank-k update W''_b u from one Hadamard-transformed group."""
    ranks = tl.arange(0, K)
    for rank in tl.static_range(0, K):
        u_rank = tl.sum(tl.where(ranks[None, :] == rank, u, 0.0), axis=1)
        w_rank = tl.load(w_h_ptr + rank * N + group * 128 + channels)
        value -= u_rank[:, None] * w_rank[None, :]
    return value


@triton.jit
def _quantize_pack_store(value, codes_ptr, scales_ptr, zeros_ptr,
                         token_offsets, group, token_mask,
                         N: tl.constexpr, BLOCK_T: tl.constexpr):
    lo = tl.min(value, axis=1)
    hi = tl.max(value, axis=1)
    raw_scale = (hi - lo) * (1.0 / 15.0)
    scale = tl.where(raw_scale > 0.0, raw_scale, 1.0).to(tl.float16)
    zero = lo.to(tl.float16)
    scale_f = tl.reshape(scale.to(tl.float32), (BLOCK_T, 1))
    zero_f = tl.reshape(zero.to(tl.float32), (BLOCK_T, 1))
    quant = tl.floor((value - zero_f) / scale_f + 0.5)
    quant = tl.maximum(0.0, tl.minimum(15.0, quant)).to(tl.uint8)
    pairs = tl.reshape(quant, (BLOCK_T, 64, 2))
    low, high = tl.split(pairs)
    packed = low | (high << 4)
    pair_offsets = tl.arange(0, 64)
    code_offsets = (
        token_offsets[:, None] * (N // 2)
        + group * 64
        + pair_offsets[None, :]
    )
    groups = N // 128
    tl.store(codes_ptr + code_offsets, packed, mask=token_mask[:, None])
    tl.store(scales_ptr + token_offsets * groups + group, scale, mask=token_mask)
    tl.store(zeros_ptr + token_offsets * groups + group, zero, mask=token_mask)


_TILE_CONFIGS = [
    triton.Config({"BLOCK_T": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 2}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 4}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_T": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 32}, num_warps=8, num_stages=2),
]


_PROJECTION_CONFIGS = [
    triton.Config({"BLOCK_T": 1, "BLOCK_D": 512}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_T": 2, "BLOCK_D": 256}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_T": 4, "BLOCK_D": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_T": 4, "BLOCK_D": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_T": 8, "BLOCK_D": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_T": 8, "BLOCK_D": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 16, "BLOCK_D": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 2, "BLOCK_D": 512}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_T": 4, "BLOCK_D": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 8, "BLOCK_D": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_T": 1, "BLOCK_D": 1024}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_PROJECTION_CONFIGS, key=["TOKENS", "N", "K", "SPLITS"])
@triton.jit
def rank_projection_kernel(x_ptr, y_ptr, partial_ptr,
                           TOKENS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           SPLITS: tl.constexpr, CHUNK: tl.constexpr,
                           BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    """u partials for BLOCK_T tokens over one of SPLITS channel chunks."""
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < TOKENS
    split = tl.program_id(1)
    ranks = tl.arange(0, K)
    lanes = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_T, K), tl.float32)
    for start in range(0, CHUNK, BLOCK_D):
        offsets = start + lanes
        channels = split * CHUNK + offsets
        # BLOCK_D need not divide CHUNK, so bound the tail against the split as
        # well as the row; masking only against N double-counts the next split.
        channel_mask = (offsets < CHUNK) & (channels < N)
        x = tl.load(
            x_ptr + token_offsets[:, None] * N + channels[None, :],
            mask=token_mask[:, None] & channel_mask[None, :], other=0.0,
        ).to(tl.float32)
        for rank in tl.static_range(0, K):
            y = tl.load(y_ptr + rank * N + channels, mask=channel_mask, other=0.0)
            column = tl.sum(x * y[None, :], axis=1)
            accumulator += tl.where(ranks[None, :] == rank, column[:, None], 0.0)
    tl.store(
        partial_ptr + token_offsets[:, None] * (SPLITS * K) + split * K + ranks[None, :],
        accumulator, mask=token_mask[:, None],
    )


@triton.autotune(configs=_TILE_CONFIGS, key=["TOKENS", "N", "K", "SPLITS"])
@triton.jit
def nar_group_quant_kernel(x_ptr, partial_ptr, w_h_ptr, codes_ptr, scales_ptr, zeros_ptr,
                           TOKENS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           SPLITS: tl.constexpr, BLOCK_T: tl.constexpr):
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    group = tl.program_id(1)
    token_mask = token_offsets < TOKENS
    channels = tl.arange(0, 128)
    x_offsets = token_offsets[:, None] * N + group * 128 + channels[None, :]
    value = tl.load(x_ptr + x_offsets, mask=token_mask[:, None], other=0.0).to(tl.float32)
    value = _fwht128(value, BLOCK_T)

    lanes = tl.arange(0, SPLITS * K)
    partial = tl.load(
        partial_ptr + token_offsets[:, None] * (SPLITS * K) + lanes[None, :],
        mask=token_mask[:, None], other=0.0,
    ).to(tl.float32)
    u = tl.sum(tl.reshape(partial, (BLOCK_T, SPLITS, K)), axis=1)
    value = _rank_correction(value, u, w_h_ptr, group, channels, N, K, BLOCK_T)
    _quantize_pack_store(
        value, codes_ptr, scales_ptr, zeros_ptr,
        token_offsets, group, token_mask, N, BLOCK_T,
    )


_FUSED_CONFIGS = [
    triton.Config({"BLOCK_T": 1, "BLOCK_D": 512}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_T": 1, "BLOCK_D": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 2, "BLOCK_D": 256}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_T": 2, "BLOCK_D": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 4, "BLOCK_D": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 4, "BLOCK_D": 128}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=_FUSED_CONFIGS, key=["TOKENS", "N", "K"])
@triton.jit
def nar_fused_row_kernel(x_ptr, y_ptr, w_h_ptr, codes_ptr, scales_ptr, zeros_ptr,
                         TOKENS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    """Variant 2: accumulate u, then re-stream the row out of L1/L2 to quantize."""
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < TOKENS
    ranks = tl.arange(0, K)
    lanes = tl.arange(0, BLOCK_D)
    u = tl.zeros((BLOCK_T, K), tl.float32)
    for start in range(0, N, BLOCK_D):
        channels = start + lanes
        channel_mask = channels < N
        x = tl.load(
            x_ptr + token_offsets[:, None] * N + channels[None, :],
            mask=token_mask[:, None] & channel_mask[None, :], other=0.0,
        ).to(tl.float32)
        for rank in tl.static_range(0, K):
            y = tl.load(y_ptr + rank * N + channels, mask=channel_mask, other=0.0)
            column = tl.sum(x * y[None, :], axis=1)
            u += tl.where(ranks[None, :] == rank, column[:, None], 0.0)
    group_channels = tl.arange(0, 128)
    for group in range(0, N // 128):
        value = tl.load(
            x_ptr + token_offsets[:, None] * N + group * 128 + group_channels[None, :],
            mask=token_mask[:, None], other=0.0,
        ).to(tl.float32)
        value = _fwht128(value, BLOCK_T)
        value = _rank_correction(value, u, w_h_ptr, group, group_channels, N, K, BLOCK_T)
        _quantize_pack_store(
            value, codes_ptr, scales_ptr, zeros_ptr,
            token_offsets, group, token_mask, N, BLOCK_T,
        )


@triton.autotune(configs=_TILE_CONFIGS, key=["TOKENS", "N"])
@triton.jit
def hadamard_group_quant_kernel(x_ptr, codes_ptr, scales_ptr, zeros_ptr,
                                TOKENS: tl.constexpr, N: tl.constexpr,
                                BLOCK_T: tl.constexpr):
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    group = tl.program_id(1)
    token_mask = token_offsets < TOKENS
    channels = tl.arange(0, 128)
    x_offsets = token_offsets[:, None] * N + group * 128 + channels[None, :]
    value = tl.load(x_ptr + x_offsets, mask=token_mask[:, None], other=0.0).to(tl.float32)
    value = _fwht128(value, BLOCK_T)
    _quantize_pack_store(
        value, codes_ptr, scales_ptr, zeros_ptr,
        token_offsets, group, token_mask, N, BLOCK_T,
    )


@dataclass
class PackedInt4:
    codes: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor


def allocate_outputs(tokens: int, n: int, device: torch.device) -> PackedInt4:
    if n % GROUP:
        raise ValueError(n)
    return PackedInt4(
        codes=torch.empty((tokens, n // 2), dtype=torch.uint8, device=device),
        scales=torch.empty((tokens, n // GROUP), dtype=torch.float16, device=device),
        zeros=torch.empty((tokens, n // GROUP), dtype=torch.float16, device=device),
    )


def split_count(tokens: int, n: int, block_t: int = 8, override: int = 0) -> int:
    """Smallest power-of-two channel split that keeps Kernel A occupancy up."""
    if override:
        if n % override:
            raise ValueError(f"{n} channels do not divide into {override} splits")
        return override
    tiles = max(1, -(-tokens // block_t))
    splits = 1
    while (splits < MAX_SPLITS and tiles * splits < MIN_PROJECTION_PROGRAMS
           and n % (2 * splits) == 0):
        splits *= 2
    return splits


def allocate_partial(tokens: int, k: int, splits: int, device: torch.device) -> torch.Tensor:
    return torch.empty((tokens, splits * k), dtype=torch.float32, device=device)


def projection_cublas(x: torch.Tensor, y_prime_bf16: torch.Tensor,
                      partial: torch.Tensor) -> torch.Tensor:
    """cuBLAS projection requested by Variant 1; output is intentionally bf16."""
    return torch.matmul(x, y_prime_bf16)


def projection_cublas_fp32(x: torch.Tensor, y_prime_fp32: torch.Tensor,
                           partial: torch.Tensor) -> torch.Tensor:
    """Numerical control for Kernel A; inputs and accumulation stay fp32."""
    return torch.matmul(x.float(), y_prime_fp32)


def projection_triton_fp32(x: torch.Tensor, y_prime_t_fp32: torch.Tensor,
                           partial: torch.Tensor) -> torch.Tensor:
    """Split streaming fp32 reduction; Kernel B folds the SPLITS partials."""
    tokens, n = x.shape
    k = y_prime_t_fp32.shape[0]
    splits = partial.shape[1] // k
    grid = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]), splits)
    rank_projection_kernel[grid](
        x, y_prime_t_fp32, partial, TOKENS=tokens, N=n, K=k, SPLITS=splits,
        CHUNK=n // splits,
    )
    return partial


def launch_nar(x_permuted: torch.Tensor, y_prime: torch.Tensor,
               w_h_t_fp32: torch.Tensor, outputs: PackedInt4, partial: torch.Tensor,
               projection: Callable[..., torch.Tensor] = projection_triton_fp32,
               ) -> torch.Tensor:
    tokens, n = x_permuted.shape
    if not x_permuted.is_contiguous():
        raise ValueError("x_permuted must be contiguous row-major")
    k = w_h_t_fp32.shape[0]
    if k not in (8, 32):
        raise ValueError(f"E17 v2 supports k=8/32, got {k}")
    u = projection(x_permuted, y_prime, partial)
    splits = u.shape[1] // k
    grid = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]), n // GROUP)
    nar_group_quant_kernel[grid](
        x_permuted, u, w_h_t_fp32,
        outputs.codes, outputs.scales, outputs.zeros,
        TOKENS=tokens, N=n, K=k, SPLITS=splits,
    )
    return u


def launch_nar_fused(x_permuted: torch.Tensor, y_prime_t_fp32: torch.Tensor,
                     w_h_t_fp32: torch.Tensor, outputs: PackedInt4) -> None:
    """Variant 2 single launch; no partial buffer and no second HBM pass."""
    tokens, n = x_permuted.shape
    if not x_permuted.is_contiguous():
        raise ValueError("x_permuted must be contiguous row-major")
    k = w_h_t_fp32.shape[0]
    grid = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]),)
    nar_fused_row_kernel[grid](
        x_permuted, y_prime_t_fp32, w_h_t_fp32,
        outputs.codes, outputs.scales, outputs.zeros,
        TOKENS=tokens, N=n, K=k,
    )


def launch_hadamard(x_permuted: torch.Tensor, outputs: PackedInt4) -> None:
    tokens, n = x_permuted.shape
    if not x_permuted.is_contiguous():
        raise ValueError("x_permuted must be contiguous row-major")
    grid = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]), n // GROUP)
    hadamard_group_quant_kernel[grid](
        x_permuted, outputs.codes, outputs.scales, outputs.zeros,
        TOKENS=tokens, N=n,
    )


def reduce_partial(u: torch.Tensor, k: int) -> torch.Tensor:
    """Host-side mirror of Kernel B's split fold, for verification only."""
    return u.float().reshape(u.shape[0], -1, k).sum(dim=1)


def unpack(codes: torch.Tensor, n: int) -> torch.Tensor:
    low = codes & 0x0F
    high = codes >> 4
    return torch.stack((low, high), dim=-1).reshape(codes.shape[0], n)


def dequantize(outputs: PackedInt4, n: int) -> torch.Tensor:
    q = unpack(outputs.codes, n).reshape(outputs.codes.shape[0], n // GROUP, GROUP).float()
    return (
        q * outputs.scales.float().unsqueeze(-1)
        + outputs.zeros.float().unsqueeze(-1)
    ).reshape(outputs.codes.shape[0], n)
