#!/usr/bin/env python3
"""Append the completed E11 fair-baseline result to report.md."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from . import experiment as base
except ImportError:
    import experiment as base


MODELS = ("llama32_3b", "llama31_8b")
MODEL_LABELS = {"llama32_3b": "Llama-3.2-3B", "llama31_8b": "Llama-3.1-8B"}
METHOD_LABELS = {
    "bf16": "bf16",
    "hadamard_g128_asym": "Hadamard, asym g128 (E5)",
    "nar_b128_kmax": "NAR, asym g128, kmax (E5)",
    "smoothquant_hadamard_g128_asym": "SmoothQuant + Hadamard, asym g128",
    "duquant_style_g128_asym": "DuQuant-style, asym g128",
    "hadamard_token_symmetric": "Hadamard, symmetric per-token",
    "hadamard_token_asymmetric": "Hadamard, asymmetric per-token",
    "nar_b64_kmax": "NAR, asym g64, kmax",
    "nar_b256_kmax": "NAR, asym g256, kmax",
    "nar_b128_k8": "NAR, asym g128, k=8",
    "nar_b128_k16": "NAR, asym g128, k=16",
    "nar_b128_k32": "NAR, asym g128, k=32",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return str(value)
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
    for model in MODELS:
        for name in ("E11_DONE.json", "e11_summary.csv", "e11_per_sequence.csv"):
            path = workdir / "results" / model / name
            if not path.exists():
                raise FileNotFoundError(path)
    decision = _json(workdir / "results" / "decision_e11.json")
    summaries = pd.concat(
        [pd.read_csv(workdir / "results" / model / "e11_summary.csv") for model in MODELS],
        ignore_index=True,
    )
    display = summaries.copy()
    display["model"] = display.model.map(MODEL_LABELS)
    display["method"] = display.method.map(METHOD_LABELS)
    display["delta_vs_Had_90CI"] = display.apply(
        lambda row: "N/A" if pd.isna(row.paired_ppl_delta_vs_hadamard) else
        f"{row.paired_ppl_delta_vs_hadamard:.6f} [{row.paired_90ci_low_vs_hadamard:.6f}, {row.paired_90ci_high_vs_hadamard:.6f}]",
        axis=1,
    )
    display["delta_vs_NAR_90CI"] = display.apply(
        lambda row: "N/A" if pd.isna(row.paired_ppl_delta_vs_nar) else
        f"{row.paired_ppl_delta_vs_nar:.6f} [{row.paired_90ci_low_vs_nar:.6f}, {row.paired_90ci_high_vs_nar:.6f}]",
        axis=1,
    )
    display = display[[
        "model", "method", "mean_ppl", "ppl_delta_vs_bf16", "effective_bits_qkv",
        "effective_bits_down", "delta_vs_Had_90CI", "delta_vs_NAR_90CI",
    ]]
    raw = pd.concat(
        [pd.read_csv(workdir / "results" / model / "e11_per_sequence.csv") for model in MODELS],
        ignore_index=True,
    )
    fold_max = float(raw.weight_fold_max_relative_error.max())
    stop = bool(decision["stop_before_e14"])
    diagnosis = (
        "The pre-registered stop condition fired: at least one fair baseline is statistically compatible with or better than NAR, so E14 must not start."
        if stop else
        "The pre-registered stop condition did not fire: neither SmoothQuant+Hadamard nor the DuQuant-style row matches NAR within the paired 90% CI on either model."
    )
    report = (workdir / "report.md").read_text()
    marker = "\n# E11 — fair baselines in the E5 setting"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    sections: list[str] = []
    sections.append("# E11 — fair baselines in the E5 setting\n")
    sections.append("The E5 bf16, both-site Hadamard, and both-site NAR rows are reused verbatim. New rows use the identical 64 WikiText-2 test chunks and three paired rotation seeds; only post-RMSNorm q/k/v inputs and down_proj inputs are fake-quantized. There was no sweep or post-result tuning.\n")
    sections.append(_table(display) + "\n")
    sections.append("Effective activation bits/value include fp16 metadata: asymmetric group-g uses 4 + 32/g bits (one fp16 scale and one fp16 real-valued zero-point per group); symmetric per-token uses 4 + 16/n; asymmetric per-token uses 4 + 32/n. SmoothQuant channel scales are statically folded into bf16 weights and therefore add no per-token metadata.\n")
    sections.append("## Baseline construction audit\n")
    sections.append("SmoothQuant uses the fixed standard alpha=0.5 rule s_c=max|x_c|^0.5/max|w_c|^0.5, applies x/s and W*s, and then the same random-sign full Hadamard.\n")
    sections.append("The DuQuant-style row was implemented after reading the [pinned official code](https://github.com/Hsu1023/DuQuant/tree/d56cfc6fe97c34c0eb100fec82fe439865905679) and the [NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/9febda1c8344cc5f2d51713964864e93-Paper-Conference.pdf). It uses the official zigzag distribution based on calibration-channel absolute maxima; within every resulting block of 128 it maps the single largest channel row to the uniform direction and uses a seeded random orthogonal basis on the complement. The official implementation can apply a greedy multi-step rotation, retain the prefix minimizing range, permute, and apply a second greedy rotation. The requested fair row deliberately omits that global multi-step prefix and second post-permutation rotation. Relative to NAR it aligns one greedy channel per block rather than top-k second-moment eigen-directions and has no explicit DC/zero-point alignment.\n")
    sections.append("For Llama-3.2-3B q/k/v, requested k=32 is dimension-capped to the 24 available group-128 DC slots; all other reported k values are realized as requested. E11 calibration used the same 128 sequences, three-pass randomized eigensolver, and stride-32 permutation-energy sample. Full eigenvalue/energy/Ritz-residual CSVs are retained; transient eigenvector checkpoints were discarded after factor construction to respect project quota.\n")
    sections.append("## Stop decision\n")
    sections.append(diagnosis + " The decision is based on baseline-minus-NAR PPL with a paired two-sided 90% Student-t CI over the three seeds; a lower bound <= 0 denotes compatible-or-better and triggers the stop.\n")
    sections.append(f"The maximum measured relative bf16 weight-fold discrepancy across the E11 rows is {fold_max:.6g}. Negative rows and engineering failures are retained.\n")
    (workdir / "report.md").write_text(report.rstrip() + "\n\n" + "\n".join(sections))
    base.atomic_json(workdir / "results" / "E11_REPORT_DONE.json", {
        "models": list(MODELS), "stop_before_e14": stop,
        "decision": decision, "max_weight_fold_relative_error": fold_max,
        "no_tuning": True,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    return parser


if __name__ == "__main__":
    build(parser().parse_args())
