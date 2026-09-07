"""Render the E22 result tables for report.md from the per-row artifacts.

Usage: python nar/e22_report_tables.py --workdir W [--write report.md]

With --write the block between ``<!-- e22-tables:start -->`` and
``<!-- e22-tables:end -->`` in the report is replaced; otherwise the
markdown is printed. Cells without an artifact print as "—".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SIZES = [("qwen3_0.6b_base", "0.6B"), ("qwen3_1.7b_base", "1.7B"), ("qwen3_4b_base", "4B"),
         ("qwen3_8b_base", "8B"), ("qwen3_14b_base", "14B")]
ROWS = [("bf16", "bf16"), ("hadamard_asym_g128", "Hadamard, W4A4KV4"),
        ("nar_k8_asym_g128", "NAR k=8"), ("nar_kmax_asym_g128", "NAR k=max")]
PPL = [("wikitext", "WikiText-2 PPL"), ("c4", "C4 PPL")]
ACC = [("mmlu", "MMLU 5-shot (acc)"), ("mmlu_redux", "MMLU-Redux 5-shot (exact match)"),
       ("gsm8k", "GSM8K 4-shot CoT (flexible extract)"), ("math", "MATH-500 4-shot CoT (exact match)"),
       ("gpqa", "GPQA-Diamond 5-shot CoT (exact match)"), ("arc_easy", "ARC-Easy 0-shot (acc_norm)"),
       ("eight_task", "Eight-task 0-shot mean")]
START, END = "<!-- e22-tables:start -->", "<!-- e22-tables:end -->"
# Qwen3 technical report, Table (Base models), 16-bit; same dict as e22_qwen3_family.TECH_REPORT.
TECH_REPORT = {
    "qwen3_0.6b_base": {"mmlu": 52.81, "mmlu_redux": 51.26, "gpqa": 26.77, "gsm8k": 59.59, "math": 32.44},
    "qwen3_1.7b_base": {"mmlu": 62.63, "mmlu_redux": 61.66, "gpqa": 28.28, "gsm8k": 75.44, "math": 43.50},
    "qwen3_4b_base": {"mmlu": 72.99, "mmlu_redux": 72.79, "gpqa": 36.87, "gsm8k": 87.79, "math": 54.10},
    "qwen3_8b_base": {"mmlu": 76.89, "mmlu_redux": 76.17, "gpqa": 44.44, "gsm8k": 89.84, "math": 60.80},
    "qwen3_14b_base": {"mmlu": 81.05, "mmlu_redux": 79.88, "gpqa": 39.90, "gsm8k": 92.49, "math": 62.02},
}


def load(workdir: Path) -> dict[tuple[str, str, str], dict]:
    out = {}
    for key, _ in SIZES:
        for path in (workdir / "results" / key).glob("e22_*_*.json"):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if "headline" in payload and "row" in payload:
                out[(key, payload["row"], payload["benchmark"])] = payload
    return out


def fmt(value, digits):
    return "—" if value is None else f"{value:.{digits}f}"


def table(data, bench, title, digits, anchor_row: bool) -> str:
    lines = [f"| {title} | " + " | ".join(label for _, label in SIZES) + " |",
             "|---|" + "---:|" * len(SIZES)]
    if anchor_row:
        anchors = [TECH_REPORT.get(key, {}).get(bench.replace("math500", "math")) for key, _ in SIZES]
        lines.append("| Qwen3 report, 16-bit | " + " | ".join(fmt(a, 2) for a in anchors) + " |")
    values = {}
    for row, label in ROWS:
        cells = []
        for key, _ in SIZES:
            v = data.get((key, row, bench), {}).get("headline")
            values[(key, row)] = v
            cells.append(fmt(v, digits))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    best_delta, had_deg, nar_deg = [], [], []
    for key, _ in SIZES:
        had = values.get((key, "hadamard_asym_g128")); k8 = values.get((key, "nar_k8_asym_g128"))
        kmax = values.get((key, "nar_kmax_asym_g128")); ref = values.get((key, "bf16"))
        nar = [v for v in (k8, kmax) if v is not None]
        if digits == 3:  # perplexity: lower is better
            best = min(nar) if nar else None
            best_delta.append(None if best is None or had is None else best - had)
            had_deg.append(None if had is None or ref is None else 100 * (had / ref - 1))
            nar_deg.append(None if best is None or ref is None else 100 * (best / ref - 1))
        else:
            best = max(nar) if nar else None
            best_delta.append(None if best is None or had is None else best - had)
    sign = lambda v, d: "—" if v is None else f"{v:+.{d}f}".replace("-", "−")
    lines.append("| NAR best − Hadamard | " + " | ".join(sign(v, digits) for v in best_delta) + " |")
    if digits == 3:
        lines.append("| Hadamard degradation | " + " | ".join("—" if v is None else f"{v:+.1f}%" for v in had_deg) + " |")
        lines.append("| NAR best degradation | " + " | ".join("—" if v is None else f"{v:+.1f}%" for v in nar_deg) + " |")
    return "\n".join(lines)


def render(workdir: Path) -> str:
    data = load(workdir)
    parts = ["## Results — perplexity", "",
             "WikiText-2 (141 windows) and C4 (256 windows of the first validation shard), 2048 tokens, fp32 NLL. "
             "Tables are generated from the artifacts by `nar/e22_report_tables.py`; a dash is a row not yet run.", ""]
    for bench, title in PPL:
        parts += [table(data, bench, title, 3, False), ""]
    parts += ["## Results — accuracy", "",
              "The Qwen3 technical report's 16-bit number is listed where it reports the benchmark; the bf16 row is this harness. "
              "A bf16 row more than 2 points from the report is a pipeline difference and is said to be one in the text.", ""]
    for bench, title in ACC:
        if bench == "eight_task" and not any(b == "eight_task" for (_, _, b) in data):
            continue
        parts += [table(data, bench, title, 2, bench in ("mmlu", "mmlu_redux", "gsm8k", "math", "gpqa")), ""]
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--write", help="report file whose marked block is replaced")
    args = ap.parse_args()
    block = render(Path(args.workdir))
    if not args.write:
        print(block)
        return
    path = Path(args.write); text = path.read_text()
    a, b = text.index(START), text.index(END)
    path.write_text(text[:a + len(START)] + "\n" + block + text[b:])
    print(f"updated {path}")


if __name__ == "__main__":
    main()
