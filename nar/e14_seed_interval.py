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
    parser.add_argument("--metric", choices=("ppl", "eight_task", "both"), default="ppl",
                        help="eight_task pairs the published eight-task mean instead "
                             "of perplexity; it needs both zero-shot artifacts per seed")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    result_dir = workdir / "results" / args.model

    def ppl(row: str, seed: int) -> float | None:
        path = result_dir / f"e14_{row}_seed{seed}_ppl.json"
        return json.loads(path.read_text())["ppl"] if path.exists() else None

    def eight_task(row: str, seed: int) -> float | None:
        """The published eight-task mean for one row and seed, or None.

        The five overlapping tasks come from the frozen artifact and the other
        three from the extra one, exactly as `finalize` forms it; a row missing
        either artifact returns None rather than a mean over what happens to be
        present, which would silently be a different statistic.
        """
        values: dict[str, float] = {}
        for suffix in ("", "_extra"):
            path = result_dir / f"e14_{row}_seed{seed}_zero_shot{suffix}.json"
            if not path.exists():
                return None
            for entry in json.loads(path.read_text())["tasks"]:
                values[entry["task"]] = float(entry["accuracy"])
        if any(task not in values for task in e14.EIGHT_TASKS):
            return None
        return float(np.mean([values[task] for task in e14.EIGHT_TASKS]))

    def interval(statistic, label: str, scale: float) -> list[dict]:
        seeds = [s for s in args.seeds if statistic("hadamard_asym_g128", s) is not None]
        missing = [s for s in args.seeds if s not in seeds]
        if missing:
            LOG.warning("%s: no Hadamard row for seeds %s; they are excluded", label, missing)
        if len(seeds) < 2:
            LOG.warning("%s: need at least two seeds for an interval, have %s", label, seeds)
        out = []
        for row in e14.PAIRED_ROWS:
            values, deltas, used = [], [], []
            for seed in seeds:
                value, had = statistic(row, seed), statistic("hadamard_asym_g128", seed)
                if value is None:
                    continue
                values.append(value * scale)
                deltas.append((value - had) * scale)
                used.append(seed)
            if not deltas:
                continue
            array = np.asarray(deltas, dtype=float)
            n = len(array)
            half = (e14.TCRIT_DF2_90 * float(array.std(ddof=1)) / math.sqrt(n)
                    if n > 1 else float("nan"))
            better = (array.mean() + half) < 0 if label == "ppl" else (array.mean() - half) > 0
            entry = {
                "model": args.model, "row": row, "statistic": label, "seeds": n,
                "seeds_used": used,
                f"{label}_mean": float(np.mean(values)),
                f"{label}_std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
                f"paired_{label}_delta_vs_hadamard": float(array.mean()),
                "paired_delta_std": float(array.std(ddof=1)) if n > 1 else 0.0,
                f"paired_{label}_ci90_low": float(array.mean() - half),
                f"paired_{label}_ci90_high": float(array.mean() + half),
                "excludes_zero": bool(n > 1 and better),
                "per_seed_delta": {str(s): float(d) for s, d in zip(used, array)},
            }
            out.append(entry)
            LOG.info("%-10s %-24s n=%d delta=%+.5f  90%% CI [%+.5f, %+.5f]  excludes zero: %s",
                     label, row, n, array.mean(), entry[f"paired_{label}_ci90_low"],
                     entry[f"paired_{label}_ci90_high"], entry["excludes_zero"])
        return out

    rows: list[dict] = []
    if args.metric in ("ppl", "both"):
        rows += interval(ppl, "ppl", 1.0)
        if rows:
            base.write_csv(result_dir / "e14_seed_interval.csv", rows)
    if args.metric in ("eight_task", "both"):
        # Percentage points, because that is the unit the published tables use.
        accuracy = interval(eight_task, "eight_task", 100.0)
        if accuracy:
            base.write_csv(result_dir / "e14_seed_interval_eight_task.csv", accuracy)
        rows += accuracy

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
