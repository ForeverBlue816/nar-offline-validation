#!/usr/bin/env python3
"""Deterministic GPU parity check against a local pinned QuaRot checkout."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch

try:
    from .quarot_gptq import GPTQ as NewGPTQ
except ImportError:
    from quarot_gptq import GPTQ as NewGPTQ


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    upstream = Path(args.upstream).resolve() / "fake_quant"
    sys.path.insert(0, str(upstream))
    # These modules are imported by QuaRot quant_utils but unused by GPTQ.
    sys.modules.setdefault("hadamard_utils", types.ModuleType("hadamard_utils"))
    sys.modules.setdefault("fast_hadamard_transform", types.ModuleType("fast_hadamard_transform"))
    from gptq_utils import GPTQ as OldGPTQ

    generator = torch.Generator(device="cpu").manual_seed(20260902)
    weight = torch.randn((17, 64), generator=generator, dtype=torch.float32).cuda()
    old_layer = torch.nn.Linear(64, 17, bias=False, device="cuda")
    new_layer = torch.nn.Linear(64, 17, bias=False, device="cuda")
    old_layer.weight.data.copy_(weight)
    new_layer.weight.data.copy_(weight)
    old = OldGPTQ(old_layer)
    old.quantizer = __import__("quant_utils").WeightQuantizer()
    old.quantizer.configure(4, perchannel=True, sym=True, mse=True)
    new = NewGPTQ(new_layer)
    for _ in range(4):
        batch = torch.randn((1, 32, 64), generator=generator).cuda()
        old.add_batch(batch, old_layer(batch))
        new.add_batch(batch)
    old.fasterquant(blocksize=128, percdamp=0.01, groupsize=-1, actorder=False, static_groups=False)
    new.fasterquant(blocksize=128, percdamp=0.01, groupsize=-1)
    difference = (old_layer.weight - new_layer.weight).abs()
    print({"max_abs_weight_error": float(difference.max()), "bitwise_equal": bool(torch.equal(old_layer.weight, new_layer.weight))})
    if not torch.equal(old_layer.weight, new_layer.weight):
        raise AssertionError("adapted GPTQ differs from pinned QuaRot")


if __name__ == "__main__":
    main()
