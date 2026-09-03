#!/usr/bin/env python3
"""Copy small result artifacts into the public tree with machine IDs redacted."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]+")


def sanitize(value: Any, key: str | None = None) -> Any:
    if key == "hostname":
        return "gpu-node"
    if isinstance(value, dict):
        return {name: sanitize(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return GPU_UUID.sub("GPU-REDACTED", value)
    return value


def publish(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial.csv"):
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            payload = sanitize(json.loads(path.read_text()))
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        elif path.suffix in {".csv", ".md", ".txt"}:
            if target.exists():
                continue
            target.write_text(path.read_text().replace("\r\n", "\n").replace("\r", "\n"))
        else:
            if target.exists():
                continue
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    publish(args.source.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
