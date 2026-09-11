#!/usr/bin/env python3
"""Export the per-layer folded NAR R4 factors (Y', W'') for the E28 integer pipeline.

Runs in the main repository environment (it needs ``nar.fold_signed_permutation``);
the E28 environment (torch 2.4 / transformers 4.36, QuaRot's e2e pins) only loads the
resulting tensor file.  One file per model:

    <workdir>/tmp/e28/nar_factors_<model>_k<k>.pt
        layers[i] = {"y_prime_fp32": (N, k), "w_h_t_fp32": (k, N)}

These are exactly the E17 v2/v3 factors: the calibrated E11 rank-k reflectors,
compact-WY'd, with the signed permutation Q=SP folded (Y' = Q W_wy, W'' = H_128 Q Y_wy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nar import activation_experiments as act  # noqa: E402
from nar import e17_v2 as v2  # noqa: E402
from nar.fold_signed_permutation import FoldedR4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--models", nargs="+", default=["llama32_3b", "llama31_8b"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    workdir = Path(args.workdir)
    out_dir = workdir / "tmp" / "e28"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    for model in args.models:
        n = int(v2.SPECS[model]["n"])
        factor_dir = v2.factor_path(workdir, model, args.rank).parent
        layers = []
        for layer in range(len(sorted(factor_dir.glob("down_layer_*.pt")))):
            factor = act.RotationFactor.load(v2.factor_path(workdir, model, args.rank, layer), device)
            folded = FoldedR4.from_factor(factor, v2.signs_for(n, layer, args.seed, device))
            assert folded.rank == args.rank, (layer, folded.rank)
            layers.append({"y_prime_fp32": folded.y_prime_fp32.contiguous(),
                           "w_h_t_fp32": folded.w_h_t_fp32.contiguous()})
        target = out_dir / f"nar_factors_{model}_k{args.rank}.pt"
        torch.save({"model": model, "n": n, "rank": args.rank, "group_size": 128,
                    "seed": args.seed, "source": str(factor_dir), "layers": layers}, target)
        print(f"{model}: {len(layers)} layers, n={n}, k={args.rank} -> {target}")


if __name__ == "__main__":
    main()
