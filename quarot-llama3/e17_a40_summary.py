#!/usr/bin/env python3
"""E28: summarise the E17 v3 kernel microbenchmark re-run on the A40.

Reads results/e28/<model>/a40_e17v3_fused_r4_timings.csv (written by
nar/e17_v3.py into the scratch workdir and copied by slurm_e28.sh) and reports,
as the E17 report does, the NAR/Hadamard transform-time ratio and the achieved
bandwidth from the byte-count model of E17 v3:

  Hadamard : read x (bf16)            + write packed INT4 codes + fp16 scale/zero per group of 128
  NAR      : Hadamard traffic + Kernel A's read of x (once per GEMM: 1 for the Triton dot,
             TERMS for the cuBLAS path) + the fp32 partial buffer (written by A, read by B)

Writes results/e28/e17v3_a40_summary.json and prints a markdown table.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "e28"
MODELS = {"3b": ("llama32_3b", 8192), "8b": ("llama31_8b", 14336)}


def bytes_moved(row: dict[str, str], n: int) -> tuple[int, int]:
    tokens, k = int(row["tokens"]), int(row["k"])
    splits = int(row["kernel_a_splits"])
    hadamard = tokens * (2 * n + n // 2 + (n // 128) * 4)
    backend = row["kernel_a_backend"]
    reads_of_x = 1
    if backend.startswith("cublas_fp32out_"):
        reads_of_x = int(backend.split("_")[2].rstrip("term"))
    partial = tokens * splits * k * 4
    nar = hadamard + reads_of_x * tokens * 2 * n + 2 * partial
    return hadamard, nar


def main() -> None:
    summary = []
    for short, (model, n) in MODELS.items():
        path = RESULTS / short / "a40_e17v3_fused_r4_timings.csv"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        for row in csv.DictReader(path.open()):
            had_bytes, nar_bytes = bytes_moved(row, n)
            had_ms, nar_ms = float(row["hadamard_fused_ms"]), float(row["nar_fused_ms"])
            summary.append({
                "model": model, "tokens": int(row["tokens"]), "k": int(row["k"]),
                "nar_fused_ms": nar_ms, "hadamard_fused_ms": had_ms, "down_matmul_ms": float(row["down_matmul_ms"]),
                "nar_over_hadamard": nar_ms / had_ms,
                "nar_over_matmul": float(row["nar_over_matmul"]), "hadamard_over_matmul": float(row["hadamard_over_matmul"]),
                "kernel_a_backend": row["kernel_a_backend"], "kernel_a_config": row["kernel_a_config"],
                "kernel_a_splits": int(row["kernel_a_splits"]), "kernel_b_nar_tile": row["kernel_b_nar_tile"],
                "hadamard_bytes": had_bytes, "nar_bytes": nar_bytes,
                "hadamard_gb_per_s": had_bytes / (had_ms * 1e-3) / 1e9,
                "nar_gb_per_s": nar_bytes / (nar_ms * 1e-3) / 1e9,
                "traffic_ratio": nar_bytes / had_bytes,
            })
    (RESULTS / "e17v3_a40_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("| tokens | model | k | NAR fused ms | Hadamard fused ms | matmul ms | NAR/Had | NAR/matmul | Had/matmul | kernel-A | splits | Had GB/s | NAR GB/s | traffic ratio |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for r in summary:
        print(f"| {r['tokens']} | {r['model']} | {r['k']} | {r['nar_fused_ms']:.6f} | {r['hadamard_fused_ms']:.6f} | "
              f"{r['down_matmul_ms']:.6f} | {r['nar_over_hadamard']:.3f} | {r['nar_over_matmul']:.3f} | "
              f"{r['hadamard_over_matmul']:.3f} | {r['kernel_a_backend']} | {r['kernel_a_splits']} | "
              f"{r['hadamard_gb_per_s']:.0f} | {r['nar_gb_per_s']:.0f} | {r['traffic_ratio']:.2f} |")


if __name__ == "__main__":
    main()
