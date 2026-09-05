"""Regression test for the fp16 metadata hazard in ``dynamic_asym_int4``.

The quantizer stores one fp16 scale and one fp16 real-valued zero per group and
documents that degenerate groups take s=1, q=0 and reproduce their fp16 offset.
The guard that implemented that tested the raw scale in fp32, which cannot see a
scale that is positive in fp32 but rounds to zero in fp16.  When that happened
the next line divided by zero; for the element equal to the group minimum, with
a minimum exactly representable in fp16, the division was 0/0 and produced a
NaN that ``clamp`` does not remove.

This was not hypothetical.  On Qwen3-8B-Base it fired on exactly one group in
24,145,920 on one evaluation window and turned every logit of that window into
NaN, taking down a whole row.  It is rare and data-dependent, which is what made
it look like an instability of the GPTQ protocol that happened to trigger it.

The fix repairs only the NaN: everywhere the old path was finite it produced
``deq = q*0 + zero = zero`` for such a group, and the repaired path produces
``q = 0``, ``deq = 0*1 + zero = zero``, the same value.  The second test pins
that, so no already-measured row needs re-running on account of this fix.

Run: python nar/quantizer_fp16_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nar import experiment as base  # noqa: E402

GROUP = 128
# fp16 rounds a value below half its smallest subnormal to zero.
FP16_ROUND_TO_ZERO = 5.960464477539063e-08 / 2


def _unguarded(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """The quantizer as it stood before the fix."""
    xg = base.group_view(x.float(), group_size)
    lo = xg.amin(dim=-1, keepdim=True)
    hi = xg.amax(dim=-1, keepdim=True)
    raw_scale = (hi - lo) / base.QMAX
    scale16 = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale)).to(torch.float16)
    zero16 = lo.to(torch.float16)
    q = torch.round((xg - zero16.float()) / scale16.float()).clamp_(0, base.QMAX)
    return (q * scale16.float() + zero16.float()).reshape_as(x)


def test_nan_group_is_repaired() -> None:
    """A group whose range underflows fp16 and whose minimum is fp16-exact."""
    x = torch.full((1, GROUP), 0.5, dtype=torch.float32)
    # 0.5 is exact in fp16; the range must leave raw_scale below the rounding
    # floor while staying a positive difference in fp32.
    x[0, 1] = 0.5 + 4e-7
    raw_scale = (x.max() - x.min()) / base.QMAX
    assert 0 < float(raw_scale) < FP16_ROUND_TO_ZERO, float(raw_scale)
    assert float(raw_scale.to(torch.float16)) == 0.0

    assert not torch.isfinite(_unguarded(x, GROUP)).all(), "the hazard no longer reproduces"
    repaired = base.dynamic_asym_int4(x, GROUP)[0]
    assert torch.isfinite(repaired).all(), "the guard did not remove the NaN"
    assert bool((repaired == 0.5).all()), "a degenerate group must reproduce its fp16 offset"


def test_fix_changes_nothing_that_was_finite() -> None:
    """Bit-identical wherever the unguarded path did not produce a NaN."""
    torch.manual_seed(0)
    for trial in range(200):
        x = torch.randn(4, 16 * GROUP)
        if trial % 3 == 1:
            x[:, :GROUP] = x[:, :1]                                  # constant groups
        if trial % 3 == 2:
            x[:, :GROUP] = x[:, :1] + torch.randn(4, GROUP) * 1e-9   # near-constant groups
        before = _unguarded(x, GROUP)
        after = base.dynamic_asym_int4(x, GROUP)[0]
        assert torch.isfinite(after).all(), f"trial {trial} still produces a non-finite value"
        finite = torch.isfinite(before)
        assert torch.equal(before[finite], after[finite]), f"trial {trial} changed a finite value"


if __name__ == "__main__":
    test_nan_group_is_repaired()
    test_fix_changes_nothing_that_was_finite()
    print("dynamic_asym_int4 fp16 regression tests pass")
