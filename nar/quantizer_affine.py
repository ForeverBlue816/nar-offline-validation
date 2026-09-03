#!/usr/bin/env python3
"""Group INT4 quantizer with an m-dimensional affine null space.

The asymmetric group quantizer already stores a real-valued zero-point per
group, which absorbs one direction for free: adding any multiple of the
group's all-ones vector leaves the integer codes unchanged.  This module
generalizes that to m directions.  For fixed orthonormal within-group
directions ``w_1 = 1/sqrt(g), w_2, ..., w_m`` (each orthogonal to the others),

    c_j = <x_g, w_j>                    j = 2..m, stored fp16
    r   = x_g - sum_{j>=2} c_j w_j
    scale = (max r - min r) / 15, zero = min r, both fp16
    q   = clamp(round((r - zero) / scale), 0, 15)
    x_hat = q * scale + zero + sum_{j>=2} c_j w_j

``m = 1`` reduces to the existing quantizer bit for bit: the DC direction keeps
its min-based zero-point, so grid utilisation is unchanged, and only the extra
directions use least-squares coefficients.

Metadata per group is 16 (scale) + 16 (zero) + 16 (m-1) bits, so the effective
width is ``4 + 16 (m + 1) / g``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from . import experiment as base
except ImportError:
    import experiment as base


QMAX = 15.0


def walsh_direction(group_size: int, index: int, device: torch.device | None = None,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sequency-ordered Walsh row ``index`` of length ``group_size``, normalized.

    index 0 is the all-ones DC vector, 1 is ``[+g/2, -g/2]`` and 2 is
    ``[+g/4, -g/4, -g/4, +g/4]``.  The rows are built directly from their sign
    pattern rather than by indexing a Hadamard matrix, so the ordering is
    sequency by construction and does not depend on the FWHT's internal order.
    """
    if group_size <= 0 or group_size & (group_size - 1):
        raise ValueError(f"group size must be a power of two, got {group_size}")
    positions = torch.arange(group_size, device=device, dtype=torch.float32)
    if index == 0:
        row = torch.ones(group_size, device=device, dtype=torch.float32)
    else:
        # Sequency k flips sign at every g / 2^k boundary, accumulated in Gray
        # order; the first two rows are the ones the protocol names explicitly.
        row = torch.ones(group_size, device=device, dtype=torch.float32)
        gray = index ^ (index >> 1)
        for bit in range(group_size.bit_length() - 1):
            if gray >> bit & 1:
                half = group_size >> (bit + 1)
                row = row * torch.where(
                    (positions // half).to(torch.int64) % 2 == 0,
                    torch.ones_like(row), -torch.ones_like(row))
    return (row / group_size ** 0.5).to(dtype)


def walsh_basis(group_size: int, m: int, device: torch.device | None = None,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The (m, g) basis w_1..w_m; verified orthonormal before it is returned."""
    if m < 1:
        raise ValueError(m)
    basis = torch.stack([walsh_direction(group_size, index, device, torch.float32)
                         for index in range(m)])
    gram = basis @ basis.T
    error = float((gram - torch.eye(m, device=gram.device)).abs().max())
    if error > 1e-5:
        raise AssertionError(f"Walsh basis is not orthonormal: max |G - I| = {error}")
    return basis.to(dtype)


@dataclass
class AffineQuant:
    dequant: torch.Tensor
    codes: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    coefficients: torch.Tensor  # (..., groups, m-1) fp16, empty when m == 1


def effective_bits(group_size: int, m: int) -> float:
    return 4.0 + 16.0 * (m + 1) / group_size


def quantize_affine(x: torch.Tensor, group_size: int, m: int = 1,
                    coefficient_dtype: torch.dtype = torch.float16) -> AffineQuant:
    """Fake-quantize with an m-dimensional affine null space per group."""
    original_dtype = x.dtype
    grouped = base.group_view(x.float(), group_size)
    if m == 1:
        residual = grouped
        coefficients = grouped.new_zeros(grouped.shape[:-1] + (0,))
        correction = torch.zeros_like(grouped)
    else:
        basis = walsh_basis(group_size, m, grouped.device)[1:]          # (m-1, g)
        raw = grouped @ basis.T                                          # (..., m-1)
        coefficients = raw.to(coefficient_dtype)
        correction = coefficients.float() @ basis
        residual = grouped - correction
    lo = residual.amin(dim=-1, keepdim=True)
    hi = residual.amax(dim=-1, keepdim=True)
    raw_scale = (hi - lo) / QMAX
    scale16 = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale)).to(torch.float16)
    zero16 = lo.to(torch.float16)
    scale = scale16.float()
    zero = zero16.float()
    codes = torch.round((residual - zero) / scale).clamp_(0, QMAX)
    dequant = codes * scale + zero + correction
    return AffineQuant(
        dequant=dequant.reshape_as(x).to(original_dtype),
        codes=codes.to(torch.uint8),
        scale=scale16.squeeze(-1),
        zero=zero16.squeeze(-1),
        coefficients=coefficients,
    )


def _self_test() -> None:
    """m=1 must reproduce the existing quantizer exactly, bit for bit."""
    torch.manual_seed(0)
    for group_size in (64, 128, 256):
        x = torch.randn(7, 4 * group_size)
        want = base.dynamic_asym_int4(x, group_size)
        got = quantize_affine(x, group_size, m=1)
        assert torch.equal(got.codes.reshape(-1), want[3].reshape(-1)), group_size
        assert torch.equal(got.scale.reshape(-1), want[1].reshape(-1)), group_size
        assert torch.equal(got.zero.reshape(-1), want[2].reshape(-1)), group_size
        assert torch.equal(got.dequant, want[0]), group_size

        basis = walsh_basis(group_size, 3)
        assert abs(float(basis[1][: group_size // 2].sum()) - group_size / 2 / group_size ** 0.5) < 1e-5
        assert abs(float(basis[1][group_size // 2:].sum()) + group_size / 2 / group_size ** 0.5) < 1e-5
        quarter = group_size // 4
        signs = [1, -1, -1, 1]
        for part, sign in enumerate(signs):
            block = basis[2][part * quarter:(part + 1) * quarter]
            assert torch.allclose(block, sign * torch.full_like(block, group_size ** -0.5)), (group_size, part)

        # Adding a null-space vector must leave the integer codes untouched.
        # That identity is exact only in exact arithmetic: c_j is stored fp16,
        # so with fp16 coefficients it holds up to that rounding. Assert the
        # exact version with fp32 coefficients, and for fp16 assert that the
        # reconstruction still tracks the added vector.
        for m in (2, 3):
            full = walsh_basis(group_size, m)
            offsets = torch.randn(7, 4, m - 1)
            delta = (offsets @ full[1:]).reshape(7, 4 * group_size)
            exact_a = quantize_affine(x, group_size, m, torch.float32)
            exact_b = quantize_affine(x + delta, group_size, m, torch.float32)
            assert torch.equal(exact_a.codes, exact_b.codes), (group_size, m)
            assert torch.equal(exact_a.scale, exact_b.scale), (group_size, m)
            assert torch.allclose(exact_b.dequant - exact_a.dequant, delta, atol=1e-4), (group_size, m)
            a = quantize_affine(x, group_size, m)
            b = quantize_affine(x + delta, group_size, m)
            drift = (b.dequant - a.dequant - delta).abs().max()
            step = a.scale.float().max()
            assert float(drift) < float(step), (group_size, m, float(drift), float(step))
            # And the m-direction quantizer must never be worse than m=1 on
            # data that actually has energy in the extra directions.
            aligned = x + (torch.randn(7, 4, m - 1) * 8.0 @ full[1:]).reshape(7, 4 * group_size)
            err_m = (quantize_affine(aligned, group_size, m).dequant - aligned).square().mean()
            err_1 = (quantize_affine(aligned, group_size, 1).dequant - aligned).square().mean()
            assert float(err_m) < float(err_1), (group_size, m, float(err_m), float(err_1))
        assert abs(effective_bits(256, 1) - 4.125) < 1e-9
        assert abs(effective_bits(256, 2) - 4.1875) < 1e-9
        assert abs(effective_bits(256, 3) - 4.25) < 1e-9
        assert abs(effective_bits(128, 1) - 4.25) < 1e-9
        assert abs(effective_bits(128, 2) - 4.375) < 1e-9
        assert abs(effective_bits(64, 1) - 4.5) < 1e-9
    print("quantizer_affine self-test passed")


if __name__ == "__main__":
    _self_test()
