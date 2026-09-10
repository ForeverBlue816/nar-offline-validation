#!/usr/bin/env python3
"""Archive finished GPTQ checkpoints to private Hugging Face repos, verify, then free the disk.

Each repo holds, for one model family:
  checkpoints/<model>/<checkpoint>/   the per-layer GPTQ state files and DONE.json
  rotations/<model>/<factor dir>/     the calibrated rotation factors the runtime hooks need
  README.md                           what the files are and how to load them

A local checkpoint directory is deleted only after every file staged for it is
present on the Hub with the same byte size.  Rotation factors are small and
stay on disk.  JSON files are uploaded with hostnames and GPU UUIDs redacted,
as nar/publish_results.py does for the public tree.

The token comes from $HF_HOME/token; it is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nar.publish_results import sanitize  # noqa: E402

WORKDIR = Path(os.environ.get("NAR_WORKDIR", "/projects/_hdd/nar/nar-validation"))
CODE_URL = "https://github.com/ForeverBlue816/nar-offline-validation"
ROTATION_DIRS = ("e14_rotations", "e14_rotations_seed1", "e14_rotations_seed2", "e11_calibration",
                 "activation_factors", "e18_factors", "e18v2_factors")

FAMILIES = {
    "llama-3.1-70b": {
        "repo": "ForeverBlue/nar-w4a4kv4-llama-3.1-70b", "experiment": "e21", "license": "llama3.1",
        "models": {"llama31_70b": "unsloth/Meta-Llama-3.1-70B"}, "section": "E21 — end-to-end W4A4KV4 on Llama-3.1-70B"},
    "llama-3.1-8b": {
        "repo": "ForeverBlue/nar-w4a4kv4-llama-3.1-8b", "experiment": "e14", "license": "llama3.1",
        "models": {"llama31_8b": "unsloth/Meta-Llama-3.1-8B"}, "section": "E14 (three seeds)"},
    "llama-3.2-3b": {
        "repo": "ForeverBlue/nar-w4a4kv4-llama-3.2-3b", "experiment": "e14", "license": "llama3.2",
        "models": {"llama32_3b": "unsloth/Llama-3.2-3B"}, "section": "E14 (weight protocols and seeds)"},
    "qwen3-base": {
        "repo": "ForeverBlue/nar-w4a4kv4-qwen3-base", "experiment": "e22", "license": "apache-2.0",
        "models": {"qwen3_0.6b_base": "Qwen/Qwen3-0.6B-Base", "qwen3_1.7b_base": "Qwen/Qwen3-1.7B-Base",
                   "qwen3_4b_base": "Qwen/Qwen3-4B-Base", "qwen3_8b_base": "Qwen/Qwen3-8B-Base"},
        "section": "E22 — the Qwen3 Base family under one protocol"},
}

PROTOCOL_TEXT = {
    "": "GPTQ default: per-channel symmetric INT4 weights (4.0 bits)",
    "act_order": "GPTQ act-order, per-channel symmetric INT4 weights (4.0 bits)",
    "g128": "GPTQ group-128 symmetric INT4 weights (4.125 bits)",
    "g128_asym": "GPTQ group-128 asymmetric INT4 weights (4.156 bits)",
}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}", flush=True)


def parse_checkpoint(name: str) -> dict[str, str]:
    """gptq_<rotation>_seed<N>[_<protocol>] -> fields."""
    body = name.removeprefix("gptq_")
    rotation, _, rest = body.partition("_seed")
    seed, _, protocol = rest.partition("_")
    return {"rotation": rotation, "seed": seed, "protocol": protocol}


def finished_checkpoints(family: dict, models: list[str] | None) -> list[tuple[str, Path]]:
    root = WORKDIR / "artifacts" / family["experiment"]
    found = []
    for model in family["models"]:
        if models and model not in models:
            continue
        for directory in sorted((root / model).glob("gptq_*")):
            if (directory / "DONE.json").exists():
                found.append((model, directory))
    return found


def stage(family_key: str, family: dict, checkpoints: list[tuple[str, Path]]) -> tuple[Path, dict[str, int]]:
    staging = WORKDIR / "tmp" / "hf_upload" / family_key
    staged: dict[str, int] = {}

    def put(source: Path, relative: str) -> None:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if source.suffix == ".json":
            target.write_text(json.dumps(sanitize(json.loads(source.read_text())), indent=2, sort_keys=True) + "\n")
        else:
            target.symlink_to(source)
        staged[relative] = target.stat().st_size

    for model, directory in checkpoints:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                put(path, f"checkpoints/{model}/{directory.name}/{path.name}")
    for model in {m for m, _ in checkpoints}:
        for name in ROTATION_DIRS:
            source = WORKDIR / "activations" / model / name
            if not source.is_dir():
                continue
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    put(path, f"rotations/{model}/{name}/{path.relative_to(source).as_posix()}")
    readme = staging / "README.md"
    readme.write_text(model_card(family, checkpoints))
    staged["README.md"] = readme.stat().st_size
    return staging, staged


def model_card(family: dict, checkpoints: list[tuple[str, Path]]) -> str:
    rows = []
    for model, directory in checkpoints:
        fields = parse_checkpoint(directory.name)
        size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
        rows.append(f"| `{model}` | `{directory.name}` | {fields['rotation']} | {fields['seed']} | "
                    f"{PROTOCOL_TEXT.get(fields['protocol'], fields['protocol'])} | {size / 1e9:.1f} GB |")
    bases = ", ".join(f"[`{b}`](https://huggingface.co/{b})" for b in family["models"].values())
    notice = ("\nBuilt with Llama. These files are derivatives of Llama 3 weights and are governed by the "
              "Llama Community License of the base model.\n") if family["license"].startswith("llama") else ""
    return f"""---
license: {family['license']}
tags: [quantization, w4a4, gptq, rotation, research-artifact]
---

# NAR W4A4KV4 GPTQ checkpoints ({', '.join(family['models'])})

Private research artifact from [nar-offline-validation]({CODE_URL}), report section "{family['section']}".
Base model(s): {bases}.
{notice}
## What the files are

`checkpoints/<model>/<checkpoint>/layer_XX.pt` holds one decoder layer's weights after the rotation
(Hadamard or NAR) has been folded into the base model and GPTQ has quantized them. The values are the
**dequantized** INT4 weights stored in floating point (fake quantization), not a packed INT4 format, so
`transformers` cannot load them on its own. `DONE.json` records the fold audit, GPTQ settings and
provenance (hostnames and GPU ids redacted). `gptq_audit.csv` has per-module GPTQ statistics.

At inference the code also quantizes activations per group of 128 and the KV cache (KIVI), and applies
the online rotations (R2 on V, R4 at the down-projection input). Those need the calibrated factors in
`rotations/<model>/`.

## Loading

Clone the code repository, download this repo so that `checkpoints/<model>/` sits at
`<workdir>/artifacts/{family['experiment']}/<model>/` and `rotations/<model>/` at
`<workdir>/activations/<model>/`, then use `nar/e14_w4a4kv4.py::load_quantized_model` (as the
experiment's `evaluate` command does).

## Checkpoints

| model | checkpoint | rotation | seed | weight protocol | size |
|---|---|---|---:|---|---:|
{chr(10).join(rows)}
"""


def remote_sizes(api, repo: str) -> dict[str, int]:
    sizes = {}
    for entry in api.list_repo_tree(repo, recursive=True, repo_type="model"):
        size = getattr(entry, "size", None)
        if size is not None and not hasattr(entry, "tree_id"):
            sizes[entry.path] = int(size)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("families", nargs="+", choices=tuple(FAMILIES))
    parser.add_argument("--models", nargs="*", help="restrict to these model keys")
    parser.add_argument("--keep", nargs="*", default=[], help="model keys whose local checkpoints must not be deleted")
    parser.add_argument("--delete", action="store_true", help="delete local checkpoints once verified")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # Measured from this cluster: Xet uploads run at 0.7 MB/s, classic LFS at
    # about 4 MB/s per stream, and eight parallel LFS streams time out against
    # S3 because the uplink is shared.  So: LFS, two workers.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi
    api = HfApi()
    manifest_path = WORKDIR / "results" / "hf_uploads.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for key in args.families:
        family = FAMILIES[key]
        checkpoints = finished_checkpoints(family, args.models)
        if not checkpoints:
            log(f"{key}: no finished checkpoints on disk, skipping")
            continue
        staging, staged = stage(key, family, checkpoints)
        total = sum(staged.values())
        log(f"{key}: staged {len(staged)} files, {total / 1e9:.1f} GB -> {family['repo']} (private)")
        api.create_repo(family["repo"], repo_type="model", private=True, exist_ok=True)
        api.upload_large_folder(repo_id=family["repo"], folder_path=staging, repo_type="model", private=True,
                                num_workers=args.workers, ignore_patterns=[".cache/**"], print_report_every=300)
        remote = remote_sizes(api, family["repo"])
        missing = sorted(p for p in staged if p not in remote)
        mismatched = sorted(p for p in staged if p in remote and remote[p] != staged[p])
        verified = not missing and not mismatched
        log(f"{key}: verification {'PASSED' if verified else 'FAILED'} "
            f"({len(staged) - len(missing) - len(mismatched)}/{len(staged)} files match; "
            f"missing {missing[:3]}, size mismatch {mismatched[:3]})")
        deleted = []
        if verified and args.delete:
            for model, directory in checkpoints:
                if model in args.keep:
                    continue
                prefix = f"checkpoints/{model}/{directory.name}/"
                if all(p in remote and remote[p] == staged[p] for p in staged if p.startswith(prefix)):
                    shutil.rmtree(directory)
                    deleted.append(str(directory.relative_to(WORKDIR)))
            log(f"{key}: deleted {len(deleted)} local checkpoint directories")
        manifest[key] = {"repo": family["repo"], "private": True, "files": len(staged), "bytes": total,
                         "verified": verified, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "checkpoints": [f"{m}/{d.name}" for m, d in checkpoints],
                         "deleted_local": deleted or manifest.get(key, {}).get("deleted_local", [])}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if verified:
            shutil.rmtree(staging / ".cache", ignore_errors=True)


if __name__ == "__main__":
    main()
