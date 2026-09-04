#!/usr/bin/env python3
"""QuaRot/GPTQ weight fake-quantization core used by E14.

This is an API-compatible, resumable adaptation of ``fake_quant/gptq_utils.py``
and ``fake_quant/quant_utils.py`` from spcl/QuaRot commit
5008669b08c1f11f9b64d52d16fddd47ca754c5a (Apache-2.0).  The numerical
choices are deliberately unchanged: per-output-channel symmetric W4, optional
MSE clipping with norm=2.4/grid=100/maxshrink=.8, block size 128, damp=.01,
one group per output channel, and no activation-order permutation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _sym_qdq(x: torch.Tensor, scale: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(x / scale), -(maxq + 1), maxq) * scale


class WeightQuantizer(torch.nn.Module):
    """QuaRot's GPTQ weight quantizer, with its clipping search unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16
        self.perchannel = True
        self.sym = True
        self.mse = False
        self.norm = 2.4
        self.grid = 100
        self.maxshrink = 0.8

    def configure(self, bits: int, perchannel: bool = True, sym: bool = True,
                  mse: bool = True, norm: float = 2.4, grid: int = 100,
                  maxshrink: float = 0.8) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.maxq = torch.tensor(2 ** (bits - 1) - 1 if sym else 2**bits - 1)

    def ready(self) -> bool:
        return bool(torch.all(self.scale != 0))

    def find_params(self, value: torch.Tensor) -> None:
        if self.bits == 16:
            return
        self.maxq = self.maxq.to(value.device)
        shape = value.shape
        x = value.flatten(1) if self.perchannel else value.flatten().unsqueeze(0)
        zeros = torch.zeros(x.shape[0], device=x.device)
        xmin = torch.minimum(x.min(1).values, zeros)
        xmax = torch.maximum(x.max(1).values, zeros)
        if not self.sym:
            raise NotImplementedError("E14 freezes QuaRot's default symmetric W4")
        xmax = torch.maximum(xmin.abs(), xmax).clamp(min=1e-5)
        self.scale = xmax / self.maxq
        self.zero = torch.zeros_like(self.scale)
        self.shrink = torch.ones(x.shape[0], device=x.device)
        if self.mse:
            best = torch.full((x.shape[0],), float("inf"), device=x.device)
            for index in range(int(self.maxshrink * self.grid)):
                shrink = 1 - index / self.grid
                scale = shrink * xmax / self.maxq
                error = _sym_qdq(x, scale.unsqueeze(1), self.maxq).sub(x).abs_().pow_(self.norm).sum(1)
                better = error < best
                if torch.any(better):
                    best[better] = error[better]
                    self.scale[better] = scale[better]
                    self.shrink[better] = shrink
        if not self.perchannel:
            self.scale = self.scale.repeat(shape[0])
            self.zero = self.zero.repeat(shape[0])
        parameter_shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(parameter_shape)
        self.zero = self.zero.reshape(parameter_shape)

    def quantize(self, value: torch.Tensor) -> torch.Tensor:
        if self.bits < 16 and self.ready():
            return _sym_qdq(value, self.scale, self.maxq).to(value.dtype)
        return value


@dataclass
class GPTQAudit:
    columns: int
    rows: int
    hessian_sequences: int
    dead_columns: int
    damp: float
    groupsize: int
    blocksize: int
    act_order: bool = False
    # Conditioning of the Hessian GPTQ inverts. A rotation that concentrates
    # activation energy onto a few input coordinates makes diag(H) spiky, and
    # because the damping is a fixed fraction of the *mean* diagonal, the same
    # spikes that raise the mean over-damp every ordinary coordinate.
    hessian_diag_max_over_median: float = 0.0
    hessian_diag_mean_over_median: float = 0.0
    damp_over_median_diag: float = 0.0
    # Share of diag(H) mass on input coordinates 0, 128, 256, ... against the
    # uninformative baseline slot_count/columns = 1/128.  This is the literal
    # "null-space slot" coordinate set, and it is the WRONG basis: the rotation
    # applies its per-group Hadamard last, so the anchor slot a reflector fills
    # becomes the group's constant direction downstream, and the Hessian GPTQ
    # builds is on the post-Hadamard activation.  Kept because it was asked for
    # and because reading it at exactly the baseline is the evidence for that.
    null_slot_diag_share: float = 0.0
    null_slot_baseline: float = 0.0
    # The same quantity in the basis GPTQ actually sees: the share of trace(H)
    # carried by the per-group all-ones direction, which is what an asymmetric
    # zero-point absorbs and what the anchor slot maps onto through H.  Same
    # 1/128 baseline, so the two are directly comparable.
    null_space_dc_share: float = 0.0
    # (tr H)^2 / ||H||_F^2, the effective number of directions the Hessian
    # spreads over. Equals the column count for an isotropic Hessian and falls
    # towards 1 as energy concentrates.
    hessian_participation_ratio: float = 0.0
    # MSE clipping is per output row here (groupsize -1 gives one scale per
    # row), so a per-input-column shrink does not exist under this protocol.
    rows_clipped: int = 0
    mean_shrink: float = 1.0
    # Of the rows that clipped, the share of their weight mass sitting on
    # null-space-slot input coordinates, against the same 1/128 baseline.
    clipped_row_null_slot_mass_share: float = 0.0
    unclipped_row_null_slot_mass_share: float = 0.0


class GPTQ:
    """The official QuaRot GPTQ update, separated from model traversal."""

    def __init__(self, layer: torch.nn.Linear) -> None:
        self.layer = layer
        self.rows, self.columns = layer.weight.shape
        self.hessian = torch.zeros((self.columns, self.columns), device=layer.weight.device)
        self.nsamples = 0
        self.quantizer = WeightQuantizer()
        self.quantizer.configure(4, perchannel=True, sym=True, mse=True)

    def add_batch(self, inp: torch.Tensor) -> None:
        # Preserve QuaRot's sequence-count normalization exactly.  A [B,T,C]
        # input contributes B to nsamples after its tokens are flattened.
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        count = inp.shape[0]
        if inp.ndim == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        matrix = inp.T
        self.hessian *= self.nsamples / (self.nsamples + count)
        self.nsamples += count
        matrix = math.sqrt(2 / self.nsamples) * matrix.float()
        self.hessian += matrix @ matrix.T

    @torch.no_grad()
    def fasterquant(self, blocksize: int = 128, percdamp: float = 0.01,
                    groupsize: int = -1, act_order: bool = False,
                    null_slot_stride: int = 128) -> GPTQAudit:
        weight = self.layer.weight.data.float().clone()
        if not self.quantizer.ready():
            self.quantizer.find_params(weight)
        hessian = self.hessian
        self.hessian = torch.empty(0, device=weight.device)
        dead = torch.diag(hessian) == 0
        dead_count = int(dead.sum())
        hessian[dead, dead] = 1
        weight[:, dead] = 0
        stats = self._diagnostics(hessian, weight, percdamp, null_slot_stride)
        if act_order:
            # QuaRot's activation ordering: quantize the highest-curvature
            # columns first, then undo the permutation at the end.
            order = torch.argsort(torch.diag(hessian), descending=True)
            weight = weight[:, order]
            hessian = hessian[order][:, order]
            inverse_order = torch.argsort(order)
        damp = float(percdamp * torch.mean(torch.diag(hessian)))
        diagonal = torch.arange(self.columns, device=weight.device)
        hessian[diagonal, diagonal] += damp
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hinv = torch.linalg.cholesky(hessian, upper=True)
        output = torch.zeros_like(weight)
        for block_start in range(0, self.columns, blocksize):
            block_stop = min(block_start + blocksize, self.columns)
            count = block_stop - block_start
            working = weight[:, block_start:block_stop].clone()
            quantized = torch.zeros_like(working)
            errors = torch.zeros_like(working)
            block_hinv = hinv[block_start:block_stop, block_start:block_stop]
            for offset in range(count):
                column = working[:, offset]
                divisor = block_hinv[offset, offset]
                if groupsize != -1 and (block_start + offset) % groupsize == 0:
                    self.quantizer.find_params(weight[:, block_start + offset:block_start + offset + groupsize])
                qcolumn = self.quantizer.quantize(column.unsqueeze(1)).flatten()
                quantized[:, offset] = qcolumn
                error = (column - qcolumn) / divisor
                working[:, offset:] -= error.unsqueeze(1) @ block_hinv[offset, offset:].unsqueeze(0)
                errors[:, offset] = error
            output[:, block_start:block_stop] = quantized
            weight[:, block_stop:] -= errors @ hinv[block_start:block_stop, block_stop:]
        torch.cuda.synchronize()
        if act_order:
            output = output[:, inverse_order]
        self.layer.weight.data.copy_(output.to(self.layer.weight.dtype))
        if torch.any(torch.isnan(self.layer.weight)):
            raise ValueError("NaN in GPTQ weights")
        return GPTQAudit(
            columns=self.columns, rows=self.rows, hessian_sequences=self.nsamples,
            dead_columns=dead_count, damp=damp, groupsize=groupsize, blocksize=blocksize,
            act_order=act_order, **stats,
        )

    @torch.no_grad()
    def _diagnostics(self, hessian: torch.Tensor, weight: torch.Tensor,
                     percdamp: float, stride: int) -> dict[str, float]:
        diag = torch.diag(hessian).float()
        median = float(diag.median())
        median = median if median > 0 else float("nan")
        slots = torch.zeros(self.columns, dtype=torch.bool, device=diag.device)
        slots[::stride] = True
        total = float(diag.sum())
        groups = self.columns // stride
        dc_share = float("nan")
        if groups * stride == self.columns and total > 0:
            # sum_g (1^T H_g 1) / |g|, the energy on each group's constant
            # direction, summed over groups and normalised by the trace.
            blocks = hessian.view(groups, stride, groups, stride)
            index = torch.arange(groups, device=diag.device)
            diagonal_blocks = blocks[index, :, index, :]
            dc_share = float(diagonal_blocks.sum(dim=(1, 2)).sum()) / (stride * total)
        frobenius = float(hessian.pow(2).sum())
        mass = weight.abs().pow(2)
        row_mass = mass.sum(1).clamp_min(1e-30)
        slot_share = (mass[:, slots].sum(1) / row_mass)
        shrink = getattr(self.quantizer, "shrink", None)
        if shrink is None:
            shrink = torch.ones(self.rows, device=diag.device)
        shrink = shrink.flatten().to(diag.device)
        clipped = shrink < 1
        return {
            "hessian_diag_max_over_median": float(diag.max()) / median,
            "hessian_diag_mean_over_median": float(diag.mean()) / median,
            "damp_over_median_diag": percdamp * float(diag.mean()) / median,
            "null_slot_diag_share": float(diag[slots].sum()) / total if total > 0 else float("nan"),
            "null_slot_baseline": float(slots.sum()) / self.columns,
            "null_space_dc_share": dc_share,
            "hessian_participation_ratio": (total * total / frobenius) if frobenius > 0 else float("nan"),
            "rows_clipped": int(clipped.sum()),
            "mean_shrink": float(shrink.mean()),
            "clipped_row_null_slot_mass_share": (
                float(slot_share[clipped].mean()) if bool(clipped.any()) else float("nan")),
            "unclipped_row_null_slot_mass_share": (
                float(slot_share[~clipped].mean()) if bool((~clipped).any()) else float("nan")),
        }
