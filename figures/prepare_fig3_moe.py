#!/usr/bin/env python3
"""Build the MoE range-law input for Figure 3(c) view 3 from the E26 per-expert table.

E26 measures, for each of the 5,342 routed experts of Qwen3-30B-A3B-Base, the
fraction f of that expert's down-input energy its own rank-k (k <= 6) R4
concentrates, and the expert's mean group-128 range under NAR divided by the
paired Hadamard range on the same held-out routed rows.  Those are exactly the
two axes of the range-law panel, measured on a second model and with the expert
rather than the layer as the unit.

They are kept in a separate file from ``fig3_range_law.csv`` on purpose: that
file defines the pooled fit drawn in every view, and it stays the three
reference sources (E1c activations, E7 V cache, E20 multi-slot, 2,912 points).
The MoE points are plotted against that same line rather than changing it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

RANGE_DESCRIPTION = ("mean_group_range / paired Hadamard mean_group_range at identical model, "
                     "site, layer, expert, and group_size")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path,
                        default=Path(os.environ.get("NAR_WORKDIR", "/projects/_hdd/nar/nar-validation")))
    parser.add_argument("--figures-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = "results/qwen3_30b_a3b_base/e26_expert_range_law.csv"
    table = pd.read_csv(args.workdir / source)
    sqrt_one_minus_f = np.sqrt(1.0 - table.f_at_k6.to_numpy())
    assert np.abs(sqrt_one_minus_f - table.predicted_sqrt_1_minus_f.to_numpy()).max() < 1e-12
    out = pd.DataFrame({
        "model": "qwen3_30b_a3b_base",
        "source_family": "E26 MoE experts",
        "source_artifact": source,
        "site": "down_input",
        "layer": table.layer,
        "expert": table.expert,
        "routed_rows": table.rows,
        "group_size": 128,
        "configuration": "nar_k6_per_expert",
        "method": "nar",
        "absorbed_energy_fraction": table.f_at_k6,
        "sqrt_one_minus_f": sqrt_one_minus_f,
        "range_ratio_vs_hadamard": table.range_nar_over_hadamard,
        "range": RANGE_DESCRIPTION,
    })
    target = args.figures_dir / "fig3_range_law_moe.csv"
    out.to_csv(target, index=False)
    print(f"{len(out)} experts -> {target}")


if __name__ == "__main__":
    main()
