#!/usr/bin/env python3
"""Append the completed E13 zero-shot transfer result to report.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from . import experiment as base
except ImportError:
    import experiment as base


MODELS = ("llama32_3b", "llama31_8b")
LABELS = {"llama32_3b": "Llama-3.2-3B", "llama31_8b": "Llama-3.1-8B"}


def _format(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (float, int)):
        return f"{value:.6g}"
    return str(value)


def _table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    frames = []
    done = []
    for model in MODELS:
        root = workdir / "results" / model
        if not (root / "E13_DONE.json").exists():
            raise FileNotFoundError(root / "E13_DONE.json")
        frames.append(pd.read_csv(root / "e13_zero_shot.csv"))
        done.append(json.loads((root / "E13_DONE.json").read_text()))
    results = pd.concat(frames, ignore_index=True)
    results["model"] = results.model.map(LABELS)
    display = results[["model", "method", "task", "metric", "accuracy", "delta_vs_bf16"]]
    report = (workdir / "report.md").read_text()
    marker = "\n# E13 — zero-shot accuracy transfer"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    verdicts = []
    for model in LABELS.values():
        subset = results[(results.model == model) & (results.task == "mean")]
        had = float(subset[subset.method == "hadamard"].accuracy.iloc[0])
        nar = float(subset[subset.method == "nar"].accuracy.iloc[0])
        verdicts.append(f"{model}: NAR-Hadamard mean-accuracy delta {nar - had:+.6f}")
    sections = [
        "# E13 — zero-shot accuracy transfer\n",
        "The pinned lm-evaluation-harness is commit b954108c9baaaa934b4ad842033b31a97ee30816. All rows are zero-shot and use seed 20260902, the same task examples, prompts, tokenizer, and metric definitions. PIQA, ARC-e, ARC-c, and HellaSwag use normalized accuracy; WinoGrande and LAMBADA use accuracy. The mean is the unweighted mean of those six values. bf16, Hadamard, and NAR use the E5 both-site activation-only setting; this one-seed transfer check has no confidence interval.\n",
        _table(display) + "\n",
        "Paired aggregate transfer: " + "; ".join(verdicts) + ".\n",
        "These accuracy results are reported regardless of sign. No task subset, prompt, batch-size, or metric was selected after observing outputs.\n",
    ]
    (workdir / "report.md").write_text(report.rstrip() + "\n\n" + "\n".join(sections))
    base.atomic_json(workdir / "results" / "E13_REPORT_DONE.json", {
        "models": list(MODELS), "source": done,
        "harness_commit": done[0]["harness_commit"], "seed": done[0]["seed"],
        "no_tuning": True,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    return parser


if __name__ == "__main__":
    build(parser().parse_args())
