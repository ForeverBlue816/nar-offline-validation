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
    "quarot_released": "QuaRot released A4 + Hadamard",
    "hadamard_asym_g128": "QuaRot Hadamard + asymmetric g128 A4",
    "nar_k8_asym_g128": "NAR k=8 R1/R4 + NAR R2 + asymmetric g128 A4",
    "nar_kmax_asym_g128": "NAR k=max R1/R4 + NAR R2 + asymmetric g128 A4",
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
    columns = ["model", "row", "seeds", "ppl", "paired_ppl_delta_vs_hadamard",
               "paired_ppl_ci90_low", "paired_ppl_ci90_high", "mean_accuracy",
               "paired_accuracy_delta_vs_hadamard", "paired_accuracy_ci90_low",
               "paired_accuracy_ci90_high", "w4_effective_bits", "a4_qkv_effective_bits",
               "a4_down_effective_bits", "k4_effective_bits_at_ctx2048",
               "v4_effective_bits_at_ctx2048", "piqa", "arc_easy", "arc_challenge",
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
        "Every row uses post-RoPE KIVI-style K4: dynamic asymmetric per-channel quantization over contiguous 32-token groups, with residual window R=32. V keeps the latest 32 tokens bf16 and quantizes older values dynamically asymmetric per token over one 128-channel head group. The Q/K rotation used by released QuaRot's per-token K path is omitted because per-channel K replaces it for every row. Hadamard rows retain random-sign R1, per-head V Hadamard plus the cross-head o_proj factor, and R4. NAR rows use calibrated global R1, per-layer per-head R2, and per-layer R4 at k=8 or k=max.\n",
        "The first row preserves upstream symmetric per-token A4 semantics while using the common KIVI K policy. The other rows quantize inputs to q/k/v/o/gate/up/down with fp16-scale/fp16-offset asymmetric group-128 A4. GPTQ uses the released seed-0 random-window sampler: 128 WikiText-2 train windows of length 2048. Rows 2–4 use three paired rotation seeds and two-sided 90% Student-t intervals over seed-level differences. Zero-shot evaluation uses batch size one so padding cannot shift token-group boundaries.\n",
        "Effective bits include metadata separately for W, A, K, and V; they are not summed across tensors. Asymmetric group-128 A4 is 4+(16+16)/128=4.25 bits/value. Cache columns include the 32-token bf16 residual at context 2048.\n",
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
