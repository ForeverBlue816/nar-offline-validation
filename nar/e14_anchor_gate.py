#!/usr/bin/env python3
"""Parse and enforce the pre-registered QuaRot Llama-2-7B PPL anchor."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


PPL_PATTERN = re.compile(r"WIKITEXT2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=float, default=6.10)
    parser.add_argument("--tolerance", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matches = PPL_PATTERN.findall(args.log.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"No WikiText-2 PPL found in {args.log}")
    actual = float(matches[-1])
    error = abs(actual - args.expected)
    passed = math.isfinite(actual) and error <= args.tolerance + 1e-12
    result = {
        "task": "E14 QuaRot release sanity anchor",
        "model": "Llama-2-7B",
        "configuration": "official QuaRot GPTQ W4A4KV4, no groups",
        "published_ppl": args.expected,
        "reproduced_ppl": actual,
        "absolute_error": error,
        "tolerance": args.tolerance,
        "passed": passed,
        "source_log": str(args.log),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
