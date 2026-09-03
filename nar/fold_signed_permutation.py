#!/usr/bin/env python3
"""Fold NAR R4's signed permutation into SwiGLU input projections.

The repository's compact-WY helper applies row vectors as
``x - (x @ wy_w) @ wy_y.T``.  In the column-vector notation used in the E17
v2 protocol this is ``G = I - W Y.T`` with ``W=wy_y`` and ``Y=wy_w``.
Keeping that distinction explicit prevents swapping the projection and
correction factors when the signed permutation is conjugated through G.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from . import activation_experiments as act
    from .e12_wy import compact_wy
except ImportError:
    import activation_experiments as act
    from e12_wy import compact_wy


def source_for_target(factor: act.RotationFactor) -> torch.Tensor:
    """Return p with ``(P x)[target] = x[p[target]]``."""
    result = torch.empty_like(factor.source_order)
    result[factor.target_order] = factor.source_order
    return result


def signed_permute_rows(value: torch.Tensor, source: torch.Tensor,
                        signs: torch.Tensor) -> torch.Tensor:
    """Apply Q=SP to a vector or to the rows of a tall matrix."""
    selected = value.index_select(0, source.to(value.device))
    shape = (signs.numel(),) + (1,) * (selected.ndim - 1)
    return selected * signs.to(value.device, value.dtype).reshape(shape)


def block_hadamard_columns(value: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Left-multiply every column by block-diagonal normalized H_group_size."""
    if value.shape[0] % group_size:
        raise ValueError((value.shape, group_size))
    transposed = value.T.contiguous().reshape(-1, value.shape[0] // group_size, group_size)
    return act.ext._fast_walsh_hadamard(transposed).reshape(value.shape[1], value.shape[0]).T


@dataclass
class FoldedR4:
    """Online H G' factors after Q=SP has been folded into gate/up weights."""

    n: int
    group_size: int
    rank: int
    source: torch.Tensor
    signs: torch.Tensor
    y_prime_fp32: torch.Tensor
    y_prime_bf16: torch.Tensor
    y_prime_t_fp32: torch.Tensor
    y_prime_lo_bf16: torch.Tensor
    y_prime_pad_bf16: torch.Tensor
    y_prime_pad_lo_bf16: torch.Tensor
    y_prime_third_bf16: torch.Tensor
    y_prime_pad_terms_bf16: torch.Tensor
    w_h_fp32: torch.Tensor
    w_h_t_fp32: torch.Tensor

    @classmethod
    def from_factor(cls, factor: act.RotationFactor, signs: torch.Tensor) -> "FoldedR4":
        if factor.b != 128:
            raise ValueError(f"E17 v2 requires group 128, got {factor.b}")
        source = source_for_target(factor)
        wy_w, wy_y = compact_wy(factor.reflectors, factor.active)
        # Column convention: G=I-WY^T has W=wy_y and Y=wy_w.
        y_prime = signed_permute_rows(wy_w, source, signs).float()
        w_prime = signed_permute_rows(wy_y, source, signs).float()
        w_h = block_hadamard_columns(w_prime, factor.b).float().contiguous()
        # tl.dot needs at least 16 columns, so Y' is zero-padded to KP; the
        # kernel stores only the first k, leaving the partial layout unchanged.
        rank = y_prime.shape[1]
        padded = max(16, 1 << (rank - 1).bit_length())
        y_bf16 = y_prime.to(torch.bfloat16)
        y_lo = (y_prime - y_bf16.float()).to(torch.bfloat16)
        pad = torch.zeros((y_prime.shape[0], padded), dtype=torch.bfloat16, device=y_prime.device)
        pad_lo = torch.zeros_like(pad)
        pad[:, :rank] = y_bf16
        pad_lo[:, :rank] = y_lo
        # Three bf16 terms recover ~24 mantissa bits for Y'; k=32 needs the
        # third to hold the fp16 zero-point inside one ULP.
        residual2 = (y_prime - y_bf16.float() - y_lo.float()).to(torch.bfloat16)
        pad_third = torch.zeros_like(pad)
        pad_third[:, :rank] = residual2
        terms = torch.cat((pad, pad_lo, pad_third), dim=0).contiguous()
        # The Triton kernels index both factors as (rank, channel) so that the
        # 128 channels of one group are contiguous under a single rank.
        return cls(
            n=factor.n,
            group_size=factor.b,
            rank=int(factor.active.sum()),
            source=source,
            signs=signs.float(),
            y_prime_fp32=y_prime,
            y_prime_bf16=y_prime.to(torch.bfloat16),
            y_prime_t_fp32=y_prime.T.contiguous(),
            y_prime_lo_bf16=y_lo,
            y_prime_pad_bf16=pad.contiguous(),
            y_prime_pad_lo_bf16=pad_lo.contiguous(),
            y_prime_third_bf16=residual2,
            y_prime_pad_terms_bf16=terms,
            w_h_fp32=w_h,
            w_h_t_fp32=w_h.T.contiguous(),
        )

    def q_unfolded(self, value: torch.Tensor) -> torch.Tensor:
        """Reference Qx used to validate the offline gate/up fold."""
        rows = value.float().reshape(-1, self.n)
        return signed_permute_rows(rows.T, self.source, self.signs).T.reshape_as(value)

    def apply(self, x_permuted: torch.Tensor) -> torch.Tensor:
        """Reference row form of H G' x_permuted."""
        shape = x_permuted.shape
        rows = x_permuted.float().reshape(-1, self.n)
        u = rows @ self.y_prime_fp32
        h = act.ext._fast_walsh_hadamard(
            rows.reshape(-1, self.n // self.group_size, self.group_size)
        ).reshape_as(rows)
        return (h - u @ self.w_h_fp32.T).reshape(shape)

    def save(self, path: Path, extra: dict | None = None) -> None:
        payload = {
            "n": self.n,
            "group_size": self.group_size,
            "rank": self.rank,
            "source_for_target": self.source.cpu(),
            "signs": self.signs.cpu(),
            "y_prime_fp32": self.y_prime_fp32.cpu(),
            "y_prime_bf16": self.y_prime_bf16.cpu(),
            "y_prime_t_fp32": self.y_prime_t_fp32.cpu(),
            "y_prime_lo_bf16": self.y_prime_lo_bf16.cpu(),
            "y_prime_pad_bf16": self.y_prime_pad_bf16.cpu(),
            "y_prime_pad_lo_bf16": self.y_prime_pad_lo_bf16.cpu(),
            "y_prime_third_bf16": self.y_prime_third_bf16.cpu(),
            "w_h_fp32": self.w_h_fp32.cpu(),
            "w_h_t_fp32": self.w_h_t_fp32.cpu(),
            "compact_wy_convention": "repository row form x-(x@wy_w)@wy_y.T; stored Y'=Q*wy_w and W''=H*Q*wy_y",
        }
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)


@torch.no_grad()
def fold_swiglu_weights(gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear,
                        folded: FoldedR4) -> None:
    """Produce Qx by row-permuting gate and signed-row-permuting up offline."""
    if gate_proj.bias is not None or up_proj.bias is not None:
        raise AssertionError("signed-permutation fold requires bias-free gate_proj/up_proj")
    if gate_proj.weight.shape[0] != folded.n or up_proj.weight.shape[0] != folded.n:
        raise ValueError((gate_proj.weight.shape, up_proj.weight.shape, folded.n))
    source = folded.source.to(gate_proj.weight.device)
    signs = folded.signs.to(up_proj.weight.device, up_proj.weight.dtype).unsqueeze(1)
    gate_proj.weight.copy_(gate_proj.weight.detach().index_select(0, source))
    up_proj.weight.copy_(up_proj.weight.detach().index_select(0, source) * signs)

def _signs(n: int, layer: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(act._seed(seed, 0, layer, "down"))
    return torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64).float().mul_(2).sub_(1).to(device)


def precompute(workdir: Path, model: str, rank: int, layers: int, seed: int) -> None:
    """Store the two online v2 factors for each layer; no dense R is materialized."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = workdir / "activations" / model / "e11_calibration" / "factors" / f"nar_b128_k{rank}"
    output = workdir / "artifacts" / "e17v2" / model / f"k{rank}"
    for layer in range(layers):
        factor = act.RotationFactor.load(root / f"down_layer_{layer:02d}.pt", device)
        folded = FoldedR4.from_factor(factor, _signs(factor.n, layer, seed, device))
        folded.save(output / f"down_layer_{layer:02d}.pt", {"model": model, "layer": layer})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rank", type=int, choices=(8, 32), required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    precompute(args.workdir.resolve(), args.model, args.rank, args.layers, args.seed)


if __name__ == "__main__":
    main()
