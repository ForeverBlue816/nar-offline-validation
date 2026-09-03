#!/usr/bin/env python3
"""E17 v3 kernels: a k-independent rank-k projection.

v2's Kernel A formed ``u = x_perm @ Y'`` with one full-width cross-lane
reduction per rank, so its cost grew linearly in k while its memory traffic (a
single read of x) did not; k=32 therefore cost 5.1x-5.4x the matched Hadamard
kernel while k=8 cost 1.85x.  v3 treats the projection as what it is, one
(N x d) @ (d x k) matmul, and runs it on tensor cores.

Kernel B, the quantizer, the fold and the verification suite are unchanged:
the group kernels below call the same ``@triton.jit`` helpers as v2.

Config selection is deliberately *not* ``triton.autotune``.  v2 shipped a
masking defect that only a fast configuration exposed, and an autotuner that
ranks on speed alone will select a wrong-but-fast config.  Here every candidate
is verified against the fp32 reference first and only verified configs are
timed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import triton
import triton.language as tl

try:
    from .r4_fused_v2 import (GROUP, PackedInt4, _fwht128, _quantize_pack_store,
                              _rank_correction, allocate_outputs, allocate_partial,
                              dequantize, reduce_partial, unpack)
except ImportError:
    from r4_fused_v2 import (GROUP, PackedInt4, _fwht128, _quantize_pack_store,
                             _rank_correction, allocate_outputs, allocate_partial,
                             dequantize, reduce_partial, unpack)


__all__ = ["GROUP", "PackedInt4", "allocate_outputs", "allocate_partial", "dequantize",
           "reduce_partial", "unpack", "ProjectionConfig", "TileConfig",
           "PROJECTION_CONFIGS", "TILE_CONFIGS", "padded_rank",
           "projection_cublas_fp32_out", "projection_cublas_fp32_out_split",
           "launch_projection_dot", "launch_nar", "launch_hadamard"]


def padded_rank(k: int) -> int:
    """tl.dot needs at least 16 columns; k=8 pads to 16, k=32 stays 32."""
    return max(16, 1 << (k - 1).bit_length())


# ------------------------------------------------------------- Option 0 ----

def _mm_fp32_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """bf16 GEMM with an fp32 output; cuBLAS already accumulates in fp32."""
    return torch.mm(x, y, out_dtype=torch.float32)


def cublas_fp32_out_supported() -> bool:
    try:
        a = torch.zeros((16, 16), dtype=torch.bfloat16, device="cuda")
        _mm_fp32_out(a, a)
        return True
    except (TypeError, RuntimeError):
        return False


def projection_cublas_fp32_out(x: torch.Tensor, y_prime_bf16: torch.Tensor,
                               partial: torch.Tensor) -> torch.Tensor:
    return _mm_fp32_out(x, y_prime_bf16)


def projection_cublas_fp32_out_split(x: torch.Tensor, terms: tuple[torch.Tensor, ...],
                                     partial: torch.Tensor) -> torch.Tensor:
    """Extra bf16 GEMMs restore mantissa bits for Y' at no extra x traffic."""
    result = _mm_fp32_out(x, terms[0])
    for term in terms[1:]:
        result.add_(_mm_fp32_out(x, term))
    return result


# ------------------------------------------------------------- Option 1 ----

@triton.jit
def rank_projection_dot_kernel(x_ptr, y_ptr, partial_ptr,
                               TOKENS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                               KP: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr,
                               TERMS: tl.constexpr,
                               BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    """u partials on tensor cores; cost is flat in k up to KP."""
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < TOKENS
    split = tl.program_id(1)
    ranks = tl.arange(0, KP)
    lanes = tl.arange(0, BLOCK_D)
    # One accumulator per bf16 term. Folding the terms into a single running
    # sum loses the third term, which is ~2^-16 of the first: it is added to an
    # accumulator already grown to the full magnitude and rounds away. Keeping
    # them separate accumulates each term among like-magnitude values and
    # rounds once at the end, which is what makes three cuBLAS GEMMs accurate;
    # here it costs registers instead of two extra passes over x.
    acc0 = tl.zeros((BLOCK_T, KP), tl.float32)
    acc1 = tl.zeros((BLOCK_T, KP), tl.float32)
    acc2 = tl.zeros((BLOCK_T, KP), tl.float32)
    for start in range(0, CHUNK, BLOCK_D):
        offsets = start + lanes
        channels = split * CHUNK + offsets
        # Bound the tail against the split as well as the row: masking only
        # against N double-counts the next split (the v2 defect).
        channel_mask = (offsets < CHUNK) & (channels < N)
        x_tile = tl.load(
            x_ptr + token_offsets[:, None] * N + channels[None, :],
            mask=token_mask[:, None] & channel_mask[None, :], other=0.0,
        )
        # Y' is carried as TERMS bf16 terms stacked along the channel axis; x
        # is already bf16 so its products are exact and only Y' is rounded.
        y0 = tl.load(y_ptr + channels[:, None] * KP + ranks[None, :],
                     mask=channel_mask[:, None], other=0.0)
        acc0 = tl.dot(x_tile, y0, acc0, out_dtype=tl.float32)
        if TERMS > 1:
            y1 = tl.load(y_ptr + N * KP + channels[:, None] * KP + ranks[None, :],
                         mask=channel_mask[:, None], other=0.0)
            acc1 = tl.dot(x_tile, y1, acc1, out_dtype=tl.float32)
        if TERMS > 2:
            y2 = tl.load(y_ptr + 2 * N * KP + channels[:, None] * KP + ranks[None, :],
                         mask=channel_mask[:, None], other=0.0)
            acc2 = tl.dot(x_tile, y2, acc2, out_dtype=tl.float32)
    accumulator = acc0
    if TERMS > 1:
        accumulator += acc1
    if TERMS > 2:
        accumulator += acc2
    tl.store(
        partial_ptr + token_offsets[:, None] * (SPLITS * K) + split * K + ranks[None, :],
        accumulator, mask=token_mask[:, None] & (ranks[None, :] < K),
    )


@dataclass(frozen=True)
class ProjectionConfig:
    block_t: int
    block_d: int
    splits: int
    num_warps: int
    num_stages: int

    def label(self) -> str:
        return (f"T{self.block_t}_D{self.block_d}_S{self.splits}"
                f"_w{self.num_warps}_s{self.num_stages}")


@dataclass(frozen=True)
class TileConfig:
    block_t: int
    num_warps: int
    num_stages: int
    use_dot: bool = False

    def label(self) -> str:
        return (f"T{self.block_t}_w{self.num_warps}_s{self.num_stages}"
                f"{'_dot' if self.use_dot else ''}")


PROJECTION_CONFIGS = tuple(
    ProjectionConfig(block_t, block_d, splits, warps, stages)
    for block_t, block_d, warps, stages in (
        (16, 64, 4, 4), (16, 128, 4, 4), (16, 256, 8, 3),
        (32, 64, 4, 4), (32, 128, 8, 3), (32, 256, 8, 3),
        (64, 64, 8, 3), (64, 128, 8, 3),
    )
    for splits in (1, 2, 4)
)

TILE_CONFIGS = tuple(
    TileConfig(block_t, warps, stages)
    for block_t, warps, stages in (
        (1, 4, 1), (2, 4, 2), (4, 4, 2), (4, 8, 2),
        (8, 4, 2), (8, 8, 3), (16, 8, 3), (32, 8, 2), (64, 8, 2),
    )
) + tuple(
    # The rank loop costs one cross-lane extraction per rank, so it grows with
    # k^2; one tl.dot replaces the whole correction. tl.dot needs BLOCK_T >= 16.
    TileConfig(block_t, warps, stages, use_dot=True)
    for block_t, warps, stages in ((16, 8, 3), (32, 8, 2), (64, 8, 2))
)


def launch_projection_dot(x: torch.Tensor, y_padded: torch.Tensor, partial: torch.Tensor,
                          k: int, config: ProjectionConfig, terms: int) -> torch.Tensor:
    tokens, n = x.shape
    kp = padded_rank(k)
    if n % config.splits:
        raise ValueError(f"{n} channels do not divide into {config.splits} splits")
    grid = (triton.cdiv(tokens, config.block_t), config.splits)
    rank_projection_dot_kernel[grid](
        x, y_padded, partial,
        TOKENS=tokens, N=n, K=k, KP=kp, SPLITS=config.splits,
        CHUNK=n // config.splits, TERMS=terms,
        BLOCK_T=config.block_t, BLOCK_D=config.block_d,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )
    return partial


# ----------------------------------------------------- Kernel B (as in v2) ---

@triton.jit
def nar_group_quant_kernel(x_ptr, partial_ptr, w_h_ptr, codes_ptr, scales_ptr, zeros_ptr,
                           TOKENS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           SPLITS: tl.constexpr, KP: tl.constexpr, USE_DOT: tl.constexpr,
                           BLOCK_T: tl.constexpr):
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
    if USE_DOT:
        ranks = tl.arange(0, KP)
        u_pad = tl.sum(tl.where((tl.arange(0, K)[None, :, None] == ranks[None, None, :]),
                                u[:, :, None], 0.0), axis=1)
        w_tile = tl.load(w_h_ptr + ranks[:, None] * N + (group * 128 + channels)[None, :],
                         mask=ranks[:, None] < K, other=0.0)
        value -= tl.dot(u_pad, w_tile, out_dtype=tl.float32, input_precision="ieee")
    else:
        value = _rank_correction(value, u, w_h_ptr, group, channels, N, K, BLOCK_T)
    _quantize_pack_store(value, codes_ptr, scales_ptr, zeros_ptr,
                         token_offsets, group, token_mask, N, BLOCK_T)


@triton.jit
def hadamard_group_quant_kernel(x_ptr, codes_ptr, scales_ptr, zeros_ptr,
                                TOKENS: tl.constexpr, N: tl.constexpr, BLOCK_T: tl.constexpr):
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    group = tl.program_id(1)
    token_mask = token_offsets < TOKENS
    channels = tl.arange(0, 128)
    x_offsets = token_offsets[:, None] * N + group * 128 + channels[None, :]
    value = tl.load(x_ptr + x_offsets, mask=token_mask[:, None], other=0.0).to(tl.float32)
    value = _fwht128(value, BLOCK_T)
    _quantize_pack_store(value, codes_ptr, scales_ptr, zeros_ptr,
                         token_offsets, group, token_mask, N, BLOCK_T)


def launch_nar(x: torch.Tensor, w_h_t_fp32: torch.Tensor, outputs: PackedInt4,
               partial: torch.Tensor, k: int, tile: TileConfig) -> None:
    tokens, n = x.shape
    splits = partial.shape[1] // k
    grid = (triton.cdiv(tokens, tile.block_t), n // GROUP)
    nar_group_quant_kernel[grid](
        x, partial, w_h_t_fp32, outputs.codes, outputs.scales, outputs.zeros,
        TOKENS=tokens, N=n, K=k, SPLITS=splits, KP=padded_rank(k),
        USE_DOT=tile.use_dot, BLOCK_T=tile.block_t,
        num_warps=tile.num_warps, num_stages=tile.num_stages,
    )


def launch_hadamard(x: torch.Tensor, outputs: PackedInt4, tile: TileConfig) -> None:
    tokens, n = x.shape
    grid = (triton.cdiv(tokens, tile.block_t), n // GROUP)
    hadamard_group_quant_kernel[grid](
        x, outputs.codes, outputs.scales, outputs.zeros,
        TOKENS=tokens, N=n, BLOCK_T=tile.block_t,
        num_warps=tile.num_warps, num_stages=tile.num_stages,
    )
