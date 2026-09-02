#!/usr/bin/env python3
"""Append completed E14 W4A4KV4 results to report.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from . import experiment as base
except ImportError:
    import experiment as base


LABELS = {
    "llama32_3b": "Llama-3.2-3B", "llama31_8b": "Llama-3.1-8B",
    "quarot": "QuaRot Hadamard + symmetric token A4",
    "hadamard_asym_g128": "QuaRot Hadamard + asymmetric g128 A4",
    "nar_asym_g128": "NAR R1/R2/R4 + asymmetric g128 A4",
}


def _table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6g}" if isinstance(value, (float, int)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    done_path = workdir / "results" / "E14_DONE.json"
    if not done_path.exists():
        raise FileNotFoundError(done_path)
    done = json.loads(done_path.read_text())
    frame = pd.read_csv(workdir / "results" / "e14_w4a4kv4_summary.csv")
    frame["model"] = frame.model.map(LABELS)
    frame["row"] = frame.row.map(LABELS)
    reference = frame[frame.row == LABELS["quarot"]].set_index("model")
    frame["ppl_delta_vs_quarot"] = frame.apply(
        lambda row: row.ppl - reference.loc[row.model].ppl, axis=1
    )
    frame["mean_acc_delta_vs_quarot"] = frame.apply(
        lambda row: row.mean_accuracy - reference.loc[row.model].mean_accuracy, axis=1
    )
    columns = ["model", "row", "ppl", "ppl_delta_vs_quarot", "mean_accuracy",
               "mean_acc_delta_vs_quarot", "piqa", "arc_easy", "arc_challenge",
               "hellaswag", "winogrande", "lambada_openai"]
    report_path = workdir / "report.md"
    report = report_path.read_text()
    marker = "\n# E14 — end-to-end W4A4KV4"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    section = [
        "# E14 — end-to-end W4A4KV4\n",
        "Weights use the GPTQ implementation and fixed clipping search from spcl/QuaRot commit 5008669b08c1f11f9b64d52d16fddd47ca754c5a: symmetric W4 per output channel, one group across the full input row, MSE norm 2.4, grid 100, max shrink 0.8, block 128, damp 0.01, no act order, and 128 fixed WikiText-2 calibration sequences. Embeddings and lm_head remain bf16 as in the upstream fake-quant path.\n",
        _table(frame[columns]) + "\n",
        "Every row uses post-RoPE KIVI-style K4: dynamic asymmetric per-channel quantization over contiguous 32-token groups, with the standard R=128 residual policy. Completed 128-token K residual chunks are quantized together, so each query sees 1..128 recent bf16 K tokens; V keeps the latest 128 tokens bf16 and quantizes older values per token over one 128-channel head group. This replaces QuaRot R3/per-token K in all rows. Hadamard rows use QuaRot's random-sign R1 and exact unsigned R2/R4; the NAR row uses calibrated global R1, per-layer per-head R2, and per-layer R4.\n",
        "The first row preserves upstream symmetric per-token A4 semantics. The other two quantize inputs to q/k/v/o/gate/up/down with fp16-scale/fp16-offset asymmetric group-128 A4. Zero-shot evaluation uses batch size one so padding cannot shift KIVI's token-group boundaries. The same seed, GPTQ calibration chunks, full contiguous WikiText-2 test stream, harness revision, prompts, tasks, and metrics are paired. This is a frozen one-seed end-to-end check, so no confidence interval is claimed.\n",
        "E12 already shows that the current unfused NAR R4 is not deployable: even a favorable quality result here does not override that engineering failure. All negative task and PPL deltas are retained.\n",
    ]
    report_path.write_text(report.rstrip() + "\n\n" + "\n".join(section))
    base.atomic_json(workdir / "results" / "E14_REPORT_DONE.json", {
        "source": done, "rows": len(frame), "no_tuning": True,
    })


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    return parser


if __name__ == "__main__":
    build(parser().parse_args())
