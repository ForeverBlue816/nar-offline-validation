"""Paired 90% interval on the E14 perplexity deltas, across seeds.

E14 was amended to a single seed, so `finalize` reports every paired delta with
a nan interval.  This computes the interval from a perplexity-only multi-seed
run, which `finalize` cannot do because it also reads a zero-shot artifact for
every seed.

The statistic is E14's own: the paired delta of a row against the
metadata-matched Hadamard row *within* each seed, then a t interval on those
deltas with `TCRIT_DF2_90`.  Pairing within a seed is what removes the
seed-to-seed drift of the model itself, which is large relative to the effect:
on Llama-3.1-8B the Hadamard row alone moves 0.015 between seeds 0 and 1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nar import e14_w4a4kv4 as e14  # noqa: E402
from nar import experiment as base  # noqa: E402

LOG = base.LOG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--model", default="llama31_8b")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    result_dir = workdir / "results" / args.model

    def ppl(row: str, seed: int) -> float | None:
        path = result_dir / f"e14_{row}_seed{seed}_ppl.json"
        return json.loads(path.read_text())["ppl"] if path.exists() else None

    seeds = [s for s in args.seeds if ppl("hadamard_asym_g128", s) is not None]
    missing = [s for s in args.seeds if s not in seeds]
    if missing:
        LOG.warning("no Hadamard row for seeds %s; they are excluded", missing)
    if len(seeds) < 2:
        LOG.warning("need at least two seeds for an interval, have %s", seeds)

    rows = []
    for row in e14.PAIRED_ROWS:
        values, deltas, used = [], [], []
        for seed in seeds:
            row_ppl, had = ppl(row, seed), ppl("hadamard_asym_g128", seed)
            if row_ppl is None:
                continue
            values.append(row_ppl)
            deltas.append(row_ppl - had)
            used.append(seed)
        if not deltas:
            continue
        array = np.asarray(deltas, dtype=float)
        n = len(array)
        if n > 1:
            half = e14.TCRIT_DF2_90 * float(array.std(ddof=1)) / math.sqrt(n)
        else:
            half = float("nan")
        entry = {
            "model": args.model, "row": row, "seeds": n, "seeds_used": used,
            "ppl_mean": float(np.mean(values)),
            "ppl_std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
            "paired_ppl_delta_vs_hadamard": float(array.mean()),
            "paired_delta_std": float(array.std(ddof=1)) if n > 1 else 0.0,
            "paired_ppl_ci90_low": float(array.mean() - half),
            "paired_ppl_ci90_high": float(array.mean() + half),
            "excludes_zero": bool(n > 1 and (array.mean() + half) < 0),
            "per_seed_delta": {str(s): float(d) for s, d in zip(used, array)},
        }
        rows.append(entry)
        LOG.info("%-24s n=%d delta=%+.5f  90%% CI [%+.5f, %+.5f]  excludes zero: %s",
                 row, n, entry["paired_ppl_delta_vs_hadamard"],
                 entry["paired_ppl_ci90_low"], entry["paired_ppl_ci90_high"],
                 entry["excludes_zero"])

    if rows:
        base.write_csv(result_dir / "e14_seed_interval.csv", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
