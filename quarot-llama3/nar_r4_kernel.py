#!/usr/bin/env python3
"""E28: the E17 v3 NAR R4 transform as a drop-in for QuaRot's ``OnlineHadamard``.

Row 3 of E28 replaces exactly one module of QuaRot's integer pipeline: the
online transform in front of the down_proj input quantizer.  QuaRot's
``OnlineHadamard(intermediate_size)`` (fast-hadamard-transform, fp16 in / fp16
out) becomes ``NARDownTransform`` (E17 v3 Kernel A + Kernel B, fp16 in / fp16
out).  The quantizer (``quarot.nn.Quantizer``, per-token symmetric INT4) and
the CUTLASS INT4 GEMM that follow are byte-identical between the two rows.

Kernel A is ``nar.kernels.r4_fused_v3.rank_projection_dot_kernel`` unchanged
(tensor-core rank-k projection ``u = x @ Y'``).  Kernel B keeps E17 v3's
structure (one program per token tile x group of 128, block-Hadamard-128 then
the rank-k correction ``H x - u W''^T``) with two changes for the A40: the
Hadamard and the correction run on tensor cores (``nar_group_transform_tc_kernel``,
see its docstring; E17's shuffle-based helpers remain in
``nar_group_transform_kernel`` for reference), and the result is stored as fp16
instead of being packed by E17's group-128 asymmetric quantizer, because
QuaRot's GEMM consumes per-token symmetric INT4 and its quantizer is kept as
the shared stage.  The E17 v3 kernels' own quantizer
epilogue is therefore not on the timed path; its cost is in E17 (Kernel B
timings) and the A40 microbenchmark of E28.

x arrives in fp16 (QuaRot's pipeline dtype), so Y' is carried as two fp16
terms (hi + lo, ~22 mantissa bits) rather than E17's two or three bf16 terms;
the per-term accumulators of Kernel A are unchanged.

Config selection follows E17 v3: every candidate is verified against the fp32
reference before it may be timed, and the fastest verified pair is selected
per token count.  ``triton.autotune`` is not used.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from nar.kernels import r4_fused_v3 as k3  # noqa: E402
from nar.kernels.r4_fused_v2 import GROUP, _fwht128, _rank_correction  # noqa: E402

TERMS = 2  # fp16 hi/lo terms for Y'
_SHARED_PARTIAL: dict[tuple[str, int, int], torch.Tensor] = {}


@triton.jit
def nar_group_transform_kernel(x_ptr, partial_ptr, w_h_ptr, out_ptr,
                               TOKENS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                               SPLITS: tl.constexpr, BLOCK_T: tl.constexpr):
    """E17 v3 Kernel B with an fp16-store epilogue (no INT4 packing)."""
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
    tl.store(out_ptr + x_offsets, value.to(tl.float16), mask=token_mask[:, None])


@triton.jit
def nar_group_transform_tc_kernel(x_ptr, partial_ptr, w_hi_ptr, w_lo_ptr, h_ptr, out_ptr,
                                  TOKENS: tl.constexpr, N: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                                  SPLITS: tl.constexpr, BLOCK_T: tl.constexpr):
    """Kernel B on tensor cores (the A40 variant used by E28).

    On sm_86 the butterfly FWHT of E17 (tl.split/join/permute shuffles) runs at
    2.4x the copy floor, whereas on the Blackwell part E17 timed it was hidden
    under the memory traffic.  Two changes, both verified against the fp32
    reference before timing:

    * H_128 is a 128x128 matrix of +-1, exact in fp16, so the block Hadamard of
      a (BLOCK_T, 128) tile is one tl.dot with fp32 accumulation and the
      1/sqrt(128) scale applied in fp32 afterwards.
    * The rank-k correction u W''^T is three (BLOCK_T, KP) @ (KP, 128) tl.dots
      on fp16 hi/lo splits of u (fp32, from Kernel A) and of W'' (hi*hi +
      lo*hi + hi*lo; the lo*lo term is ~2^-22 and dropped), instead of E17's
      per-rank cross-lane extraction loop.  u and W'' keep ~22 mantissa bits.
    """
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    group = tl.program_id(1)
    token_mask = token_offsets < TOKENS
    channels = tl.arange(0, 128)
    x_offsets = token_offsets[:, None] * N + group * 128 + channels[None, :]
    x_tile = tl.load(x_ptr + x_offsets, mask=token_mask[:, None], other=0.0)
    h_tile = tl.load(h_ptr + channels[:, None] * 128 + channels[None, :])
    value = tl.dot(x_tile, h_tile, out_dtype=tl.float32) * 0.08838834764831845
    ranks = tl.arange(0, KP)
    rank_mask = ranks < K
    u = tl.zeros((BLOCK_T, KP), tl.float32)
    for split in tl.static_range(SPLITS):
        u += tl.load(partial_ptr + token_offsets[:, None] * (SPLITS * K) + split * K + ranks[None, :],
                     mask=token_mask[:, None] & rank_mask[None, :], other=0.0)
    u_hi = u.to(tl.float16)
    u_lo = (u - u_hi.to(tl.float32)).to(tl.float16)
    w_offsets = ranks[:, None] * N + group * 128 + channels[None, :]
    w_hi = tl.load(w_hi_ptr + w_offsets, mask=rank_mask[:, None], other=0.0)
    w_lo = tl.load(w_lo_ptr + w_offsets, mask=rank_mask[:, None], other=0.0)
    correction = tl.dot(u_hi, w_hi, out_dtype=tl.float32)
    correction = tl.dot(u_lo, w_hi, correction, out_dtype=tl.float32)
    correction = tl.dot(u_hi, w_lo, correction, out_dtype=tl.float32)
    value -= correction
    tl.store(out_ptr + x_offsets, value.to(tl.float16), mask=token_mask[:, None])


@dataclass(frozen=True)
class TileConfig:
    block_t: int
    num_warps: int
    num_stages: int
    use_tc: bool = False

    def label(self) -> str:
        return f"T{self.block_t}_w{self.num_warps}_s{self.num_stages}{'_tc' if self.use_tc else ''}"


SHUFFLE_TILE_CONFIGS = tuple(TileConfig(*c) for c in (
    (1, 4, 1), (2, 4, 2), (4, 4, 2), (4, 8, 2), (8, 4, 2), (8, 8, 3), (16, 8, 3), (32, 8, 2)))
TILE_CONFIGS = tuple(TileConfig(block_t, warps, 2, True) for block_t, warps in (
    (16, 4), (32, 4), (64, 4), (16, 8), (32, 8), (64, 8), (128, 8)))
PROJECTION_CONFIGS = k3.PROJECTION_CONFIGS
REFERENCE_TILE = TileConfig(32, 4, 2, True)


def sylvester_hadamard_fp16(device: torch.device) -> torch.Tensor:
    """H_128 with entries +-1 in the natural (butterfly) order that _fwht128 uses."""
    h = torch.ones((1, 1), dtype=torch.float32)
    core = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    while h.shape[0] < GROUP:
        h = torch.kron(core, h)
    return h.to(device, torch.float16).contiguous()


def w_terms(w_h_t_fp32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """W''^T (k, N) fp32 -> fp16 hi and lo terms (k, N) each."""
    hi = w_h_t_fp32.to(torch.float16).contiguous()
    lo = (w_h_t_fp32 - hi.float()).to(torch.float16).contiguous()
    return hi, lo


def launch_kernel_b(x: torch.Tensor, partial: torch.Tensor, w_h_t_fp32: torch.Tensor | None,
                    out: torch.Tensor, k: int, tile: TileConfig, h128: torch.Tensor | None = None,
                    w_hi: torch.Tensor | None = None, w_lo: torch.Tensor | None = None) -> None:
    tokens, n = x.shape
    splits = partial.shape[1] // k
    grid = (triton.cdiv(tokens, tile.block_t), n // GROUP)
    if tile.use_tc:
        assert h128 is not None and w_hi is not None and w_lo is not None
        nar_group_transform_tc_kernel[grid](
            x, partial, w_hi, w_lo, h128, out,
            TOKENS=tokens, N=n, K=k, KP=k3.padded_rank(k), SPLITS=splits, BLOCK_T=tile.block_t,
            num_warps=tile.num_warps, num_stages=tile.num_stages)
        return
    assert w_h_t_fp32 is not None
    nar_group_transform_kernel[grid](
        x, partial, w_h_t_fp32, out,
        TOKENS=tokens, N=n, K=k, SPLITS=splits, BLOCK_T=tile.block_t,
        num_warps=tile.num_warps, num_stages=tile.num_stages)


def fp16_terms(y_prime_fp32: torch.Tensor, k: int) -> torch.Tensor:
    """Y' (N, k) -> TERMS stacked fp16 blocks of shape (N, KP) as Kernel A reads them."""
    n = y_prime_fp32.shape[0]
    kp = k3.padded_rank(k)
    hi = y_prime_fp32.to(torch.float16)
    lo = (y_prime_fp32 - hi.float()).to(torch.float16)
    terms = torch.zeros((TERMS, n, kp), dtype=torch.float16, device=y_prime_fp32.device)
    terms[0, :, :k] = hi
    terms[1, :, :k] = lo
    return terms.reshape(TERMS * n, kp).contiguous()


def reference_transform(x: torch.Tensor, y_prime_fp32: torch.Tensor, w_h_t_fp32: torch.Tensor) -> torch.Tensor:
    """fp32 reference of H_128 x - (x Y') W''^T (E17's FoldedR4.apply without the fold)."""
    rows = x.float()
    n = rows.shape[1]
    h = rows.reshape(-1, n // GROUP, GROUP)
    width = 1
    while width < GROUP:
        blocks = h.reshape(-1, GROUP // (2 * width), 2, width)
        h = torch.cat((blocks[:, :, 0, :] + blocks[:, :, 1, :], blocks[:, :, 0, :] - blocks[:, :, 1, :]), dim=-1)
        width *= 2
    h = h.reshape(-1, n) / math.sqrt(GROUP)
    u = rows @ y_prime_fp32
    return h - u @ w_h_t_fp32


def quarot_codes(y: torch.Tensor) -> torch.Tensor:
    """QuaRot's per-token symmetric INT4 codes (what quarot.nn.Quantizer produces)."""
    scale = (y.abs().amax(dim=-1, keepdim=True).to(torch.float16).float() / 7.0)
    return torch.clamp(torch.round(y.float() / scale.clamp_min(1e-8)), -8, 7)


def precision_row(observed_fp16: torch.Tensor, reference_fp32: torch.Tensor) -> dict[str, float]:
    diff = observed_fp16.float() - reference_fp32
    codes_obs = quarot_codes(observed_fp16.float())
    codes_ref = quarot_codes(reference_fp32)
    return {
        "relative_l2": float(diff.norm() / reference_fp32.norm().clamp_min(1e-30)),
        "max_abs_over_row_absmax": float((diff.abs().amax(dim=1) / reference_fp32.abs().amax(dim=1)).max()),
        "code_match_fraction": float((codes_obs == codes_ref).float().mean()),
    }


class NARDownTransform(torch.nn.Module):
    """fp16 (..., N) -> fp16 (..., N): H_128 G' applied to the (offline-permuted) down_proj input."""

    def __init__(self, y_prime_fp32: torch.Tensor, w_h_t_fp32: torch.Tensor, k: int,
                 selection: dict[str, Any], h128: torch.Tensor | None = None):
        super().__init__()
        self.n = int(y_prime_fp32.shape[0])
        self.k = k
        self.register_buffer("y_terms", fp16_terms(y_prime_fp32, k))
        w_hi, w_lo = w_terms(w_h_t_fp32)
        self.register_buffer("w_hi", w_hi)
        self.register_buffer("w_lo", w_lo)
        # H_128 (+-1, fp16, 32 KB) is one tensor shared by every layer's module.
        self.register_buffer("h128", h128 if h128 is not None else sylvester_hadamard_fp16(y_prime_fp32.device))
        self.selection = selection  # {"tokens": {"proj": ProjectionConfig, "tile": TileConfig}}
        self._plans = {int(t): (k3.ProjectionConfig(**s["proj"]), TileConfig(**s["tile"]))
                       for t, s in selection.items()}
        if any(not tile.use_tc for _, tile in self._plans.values()):
            self.register_buffer("w_h_t", w_h_t_fp32.contiguous())
        else:
            self.w_h_t = None
        self._compiled: dict[int, Any] = {}

    def factor_bytes(self) -> int:
        """Per-layer bytes the NAR transform adds to the model (H_128 is shared, counted once elsewhere)."""
        extra = self.w_h_t.numel() * 4 if self.w_h_t is not None else 0
        return self.y_terms.numel() * 2 + self.w_hi.numel() * 2 + self.w_lo.numel() * 2 + extra

    def plan(self, tokens: int) -> tuple[k3.ProjectionConfig, TileConfig]:
        if tokens in self._plans:
            return self._plans[tokens]
        nearest = min(self._plans, key=lambda t: abs(math.log(t) - math.log(tokens)))
        return self._plans[nearest]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.n)
        if not rows.is_contiguous():
            rows = rows.contiguous()
        tokens = rows.shape[0]
        fast = self._compiled.get(tokens)
        if fast is None:
            fast = self._compile(tokens)
        if fast is not False:
            run_a, run_b, partial = fast
            out = torch.empty_like(rows)
            run_a(rows, self.y_terms, partial)
            run_b(rows, partial, self.w_hi, self.w_lo, self.h128, out)
            return out.view(x.shape)
        proj, tile = self.plan(tokens)
        partial = torch.empty((tokens, proj.splits * self.k), dtype=torch.float32, device=rows.device)
        k3.launch_projection_dot(rows, self.y_terms, partial, self.k, proj, TERMS)
        out = torch.empty_like(rows)
        launch_kernel_b(rows, partial, self.w_h_t, out, self.k, tile, self.h128, self.w_hi, self.w_lo)
        return out.view(x.shape)

    def _compile(self, tokens: int):
        """Pre-bound launchers for one token count (decode is launch-bound).

        Triton's ``kernel[grid](...)`` re-binds and re-specialises its arguments on
        every call (~40 us of Python per launch on this host); QuaRot's stages are
        C++ extension ops with ~10 us launches.  The kernels are compiled once per
        token count with ``JITFunction.warmup`` and launched through the
        ``CompiledKernel`` runner, which skips the binding.  The kernels, configs
        and outputs are identical to the generic path (checked here bit-for-bit);
        the partial buffer is reused across calls (the stream is sequential).
        """
        proj, tile = self.plan(tokens)
        if not tile.use_tc:
            self._compiled[tokens] = False
            return False
        device = self.y_terms.device
        n, k = self.n, self.k
        kp = k3.padded_rank(k)
        try:
            x = (torch.randn((tokens, n), device=device) * 0.5).to(torch.float16)
            # The fp32 partial buffer is transient (written by A, consumed by B in
            # the same forward), so one buffer per (device, shape) is shared by all
            # layers instead of one per layer: 28 x 1 MB at 32768 tokens otherwise
            # shows up as resident memory that is not part of the factors.
            key = (str(device), tokens, proj.splits * k)
            partial = _SHARED_PARTIAL.get(key)
            if partial is None:
                partial = torch.empty((tokens, proj.splits * k), dtype=torch.float32, device=device)
                _SHARED_PARTIAL[key] = partial
            out = torch.empty_like(x)
            grid_a = (triton.cdiv(tokens, proj.block_t), proj.splits, 1)
            grid_b = (triton.cdiv(tokens, tile.block_t), n // GROUP, 1)
            kernel_a = k3.rank_projection_dot_kernel.warmup(
                x, self.y_terms, partial, TOKENS=tokens, N=n, K=k, KP=kp, SPLITS=proj.splits,
                CHUNK=n // proj.splits, TERMS=TERMS, BLOCK_T=proj.block_t, BLOCK_D=proj.block_d,
                num_warps=proj.num_warps, num_stages=proj.num_stages, grid=grid_a)
            kernel_b = nar_group_transform_tc_kernel.warmup(
                x, partial, self.w_hi, self.w_lo, self.h128, out, TOKENS=tokens, N=n, K=k, KP=kp,
                SPLITS=proj.splits, BLOCK_T=tile.block_t, num_warps=tile.num_warps,
                num_stages=tile.num_stages, grid=grid_b)
            run_a, run_b = kernel_a[grid_a], kernel_b[grid_b]
            run_a(x, self.y_terms, partial)
            run_b(x, partial, self.w_hi, self.w_lo, self.h128, out)
            reference_partial = torch.empty_like(partial)
            reference = torch.empty_like(out)
            k3.launch_projection_dot(x, self.y_terms, reference_partial, k, proj, TERMS)
            launch_kernel_b(x, reference_partial, None, reference, k, tile, self.h128, self.w_hi, self.w_lo)
            torch.cuda.synchronize()
            if not torch.equal(out, reference):
                raise RuntimeError("pre-bound launch differs from the generic launch")
            self._compiled[tokens] = (run_a, run_b, partial)
        except Exception as error:  # noqa: BLE001 - fall back to the generic launch path
            import warnings
            warnings.warn(f"NARDownTransform: pre-bound launch unavailable for {tokens} tokens ({error}); "
                          "using the generic Triton launch path")
            self._compiled[tokens] = False
            return False
        return self._compiled[tokens]


# ------------------------------------------------------------ selection ----

def _bench(fn, warmup: int = 10, rep: int = 50) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def select_configs(y_prime_fp32: torch.Tensor, w_h_t_fp32: torch.Tensor, k: int,
                   token_counts: list[int], seed: int, tolerance: dict[str, float],
                   verify_rows: int = 256) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify every (Kernel A config, Kernel B tile) then time the verified ones per token count."""
    device = y_prime_fp32.device
    n = int(y_prime_fp32.shape[0])
    terms = fp16_terms(y_prime_fp32, k)
    h128 = sylvester_hadamard_fp16(device)
    w_hi, w_lo = w_terms(w_h_t_fp32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample = (torch.randn((verify_rows, n), generator=generator) * 0.5).to(device, torch.float16)
    reference = reference_transform(sample, y_prime_fp32, w_h_t_fp32)
    log: list[dict[str, Any]] = []

    def run(x: torch.Tensor, proj: k3.ProjectionConfig, tile: TileConfig) -> torch.Tensor:
        partial = torch.empty((x.shape[0], proj.splits * k), dtype=torch.float32, device=device)
        k3.launch_projection_dot(x, terms, partial, k, proj, TERMS)
        out = torch.empty_like(x)
        launch_kernel_b(x, partial, w_h_t_fp32, out, k, tile, h128, w_hi, w_lo)
        return out

    def passes(row: dict[str, float]) -> bool:
        return (row["code_match_fraction"] >= tolerance["code_match_fraction_min"]
                and row["relative_l2"] <= tolerance["relative_l2_max"])

    verified_proj: list[k3.ProjectionConfig] = []
    for proj in PROJECTION_CONFIGS:
        if n % proj.splits:
            continue
        try:
            out = run(sample, proj, REFERENCE_TILE)
            torch.cuda.synchronize()
            row = precision_row(out, reference)
        except Exception as error:  # noqa: BLE001 - a config that cannot run cannot be selected
            row = {"error": str(error)[:200], "code_match_fraction": 0.0, "relative_l2": float("inf")}
        row.update({"stage": "kernel_a", "config": proj.label(), "passes": passes(row)})
        log.append(row)
        if row["passes"]:
            verified_proj.append(proj)
    if not verified_proj:
        raise AssertionError("no Kernel A configuration passes verification")
    verified_tiles: list[TileConfig] = []
    for tile in TILE_CONFIGS:
        try:
            out = run(sample, verified_proj[0], tile)
            torch.cuda.synchronize()
            row = precision_row(out, reference)
        except Exception as error:  # noqa: BLE001
            row = {"error": str(error)[:200], "code_match_fraction": 0.0, "relative_l2": float("inf")}
        row.update({"stage": "kernel_b", "config": tile.label(), "passes": passes(row)})
        log.append(row)
        if row["passes"]:
            verified_tiles.append(tile)
    if not verified_tiles:
        raise AssertionError("no Kernel B tile passes verification")

    selection: dict[str, Any] = {}
    for tokens in token_counts:
        x = (torch.randn((tokens, n), generator=generator) * 0.5).to(device, torch.float16)
        best = None
        # Kernel A's split count sets Kernel B's partial width, so select the pair
        # on the combined pipeline (E17 v3 protocol).
        for proj in verified_proj:
            for tile in verified_tiles:
                ms = _bench(lambda p=proj, t=tile: run(x, p, t))
                log.append({"stage": "timing", "tokens": tokens, "kernel_a": proj.label(),
                            "kernel_b": tile.label(), "ms": ms})
                if best is None or ms < best[0]:
                    best = (ms, proj, tile)
        assert best is not None
        ms, proj, tile = best
        selection[str(tokens)] = {"proj": proj.__dict__, "tile": tile.__dict__, "pipeline_ms": ms,
                                  "kernel_a_ms": _bench(lambda: k3.launch_projection_dot(
                                      x, terms, torch.empty((tokens, proj.splits * k), dtype=torch.float32,
                                                            device=device), k, proj, TERMS))}
        del x
    return selection, log


def load_factors(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["layers"] = [{key: value.to(device) for key, value in layer.items()} for layer in payload["layers"]]
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
