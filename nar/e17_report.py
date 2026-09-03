#!/usr/bin/env python3
"""Append the completed strict one-read E17 result to report.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(workdir: Path) -> None:
    result = json.loads((workdir / "results" / "llama32_3b" / "E17_DONE.json").read_text())
    report_path = workdir / "report.md"
    report = report_path.read_text()
    marker = "\n# E17 — fused one-pass R4"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    verify = result["verification"]
    lines = [
        "# E17 — fused one-pass R4\n",
        "The final kernel is a literal one-read implementation: each bf16 token row is loaded once, the rank-8 compact-WY projections are formed, the frozen permutation is performed with a Triton register gather, and signs, block-H128, dynamic asymmetric group-128 INT4 quantization, and INT4 packing are completed in the same kernel launch. It emits two INT4 codes per uint8 plus one fp16 scale and one fp16 real-valued zero per group. The matched Hadamard kernel fuses signs, block-H128, and exactly the same quantizer/packer.\n",
        f"Verification is exact on {verify['verify_tokens']} frozen random token rows: both methods have code-match fraction 1.0 and zero max error in packed codes, fp16 scales, fp16 zero-points, and dequantized values versus the PyTorch reference (allowed bf16 tolerance {verify['bf16_tolerance']}).\n",
        "| tokens | NAR fused ms | Hadamard fused ms | down_proj bf16 ms | NAR / Hadamard | NAR / matmul | transform FLOP / matmul |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["timings"]:
        lines.append(
            f"| {row['tokens']} | {row['nar_fused_ms']:.6f} | {row['hadamard_fused_ms']:.6f} | "
            f"{row['down_matmul_ms']:.6f} | {row['nar_overhead_vs_hadamard_ratio']:.3f}× | "
            f"{row['nar_ratio_vs_down_matmul']:.3f}× | {row['nar_transform_flop_ratio_vs_matmul']:.6f} |"
        )
    lines.extend([
        "\n**2048-token engineering gate: FAIL.** The strict fused NAR kernel costs 4.225× the down_proj matmul and 29.286× the matched fused Hadamard kernel, far above the 10% limit. E12 is therefore **not superseded**. The arithmetic count is only 0.635% of the matmul FLOPs, but the one-program-per-token global reductions plus an arbitrary 8192-element register gather create severe register pressure and low occupancy; this implementation is bandwidth/compiler-scheduling bound rather than FLOP bound. At one token both transforms are also launch-bound, but NAR remains 5.540× the matmul versus 0.419× for fused Hadamard. This negative deployability result is retained without tuning.\n",
        "Exact timings and verification metadata are in `results/llama32_3b/e17_fused_r4_timings.csv` and `E17_DONE.json`.\n",
    ])
    report_path.write_text(report.rstrip() + "\n\n" + "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    build(args.workdir.resolve())


if __name__ == "__main__":
    main()
