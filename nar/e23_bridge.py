"""E23 — our rows under TwinQuant's protocol (arXiv 2606.01556).

One table where our rows and TwinQuant's published Qwen3 W4A4 rows share a
protocol. The protocol is theirs; this module re-points the E22 pipeline at
it and changes nothing else:

  weights      GPTQ, group-128 symmetric (E14 protocol ``g128``, 4.125 bits)
               -- "each contiguous group of 128 elements shares one
               quantization scale" (TwinQuant 5.1), symmetric per their Eq. 1
  activations  group-128 symmetric int4, dynamic, every linear input
               (``symmetric_g128``, 4.125 bits) -- same sentence
  KV cache     not quantized (the paper never mentions the cache): W4A4KV16
  calibration  128 x 2048 WikiText-2 sequences (stated; E14's default)
  context      2048 (unstated in the paper; the convention their calibration
               uses and the one every other W4A4 table uses)
  checkpoints  the post-trained Qwen/Qwen3-8B, -14B, -32B (unstated in the
               paper; Qwen3-32B exists only post-trained, and the printed
               Qwen3-8B per-task row is the post-trained profile: PIQA 76.4,
               WinoGrande 68.0, LAMBADA 67.4 against the Base model's
               79.3 / 72.8 / 72.2 measured in E19)
  zero-shot    their six tasks (ARC-c, ARC-e, HellaSwag, LAMBADA, PIQA,
               WinoGrande), zero-shot, this repository's pinned harness
               (their version is unstated); acc_norm where defined
  perplexity   WikiText-2 test, 2048-token windows, fp32 NLL (E14/E19/E22
               convention; their chunking and BOS handling are unstated,
               and their 16-bit row is listed next to ours to show the gap)

Everything that is an assumption is recorded in results/e23_protocol.json and
in every artifact's provenance. Artifacts: results/<model>/e23_<row>_<bench>.json,
checkpoints under artifacts/e23/. ``bridge`` writes results/e23_bridge_summary.csv
with TwinQuant's rows beside ours.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nar import e14_w4a4kv4 as e14  # noqa: E402
from nar import e19_qwen3_e2e as e19  # noqa: E402
from nar import e21_llama70b_e2e as e21  # noqa: E402
from nar import e22_qwen3_family as e22  # noqa: E402

E23_FAMILY = ("qwen3_8b", "qwen3_14b", "qwen3_32b")
BENCHMARKS = ("wikitext", "six_task")

# TwinQuant Table 5 (appendix), Qwen3 rows, verbatim: per task ARC-C ARC-E
# HellaSwag PIQA Winogrande LAMBADA, then the six-task mean and WikiText-2 PPL.
TASKS = ("ARC-C", "ARC-E", "HellaSwag", "PIQA", "Winogrande", "LAMBADA")
TWINQUANT = {
    "qwen3_8b": {
        "W16A16": (55.5, 83.5, 78.8, 76.4, 68.0, 67.4, 71.6, 9.71),
        "RTN": (22.6, 24.9, 45.6, 48.9, 51.8, 46.9, 40.1, 4392),
        "SpinQuant": (53.6, 78.5, 71.4, 76.5, 67.3, 62.4, 68.3, 14.8),
        "QuaRot": (49.8, 74.8, 69.8, 68.3, 60.7, 57.1, 63.4, 24.5),
        "SmoothQuant": (25.7, 25.5, 41.7, 50.5, 52.2, 44.9, 40.1, 3360.1),
        "FlatQuant": (54.4, 79.4, 73.1, 77.0, 68.8, 63.4, 69.3, 13.4),
        "SVDQuant": (52.8, 78.9, 74.4, 75.8, 66.6, 64.1, 68.8, 14.9),
        "TwinQuant": (53.6, 80.8, 77.1, 75.7, 67.9, 65.8, 70.2, 13.2),
    },
    "qwen3_14b": {
        "W16A16": (59.0, 84.3, 80.5, 80.0, 72.9, 68.4, 74.2, 8.6),
        "RTN": (24.8, 23.9, 50.6, 55.7, 46.8, 50.7, 42.1, 18749),
        "SpinQuant": (56.3, 81.0, 75.8, 76.9, 71.4, 65.7, 71.2, 13.0),
        "QuaRot": (52.8, 76.4, 72.4, 73.6, 65.0, 62.8, 67.2, 18.2),
        "SmoothQuant": (26.5, 25.8, 48.7, 51.2, 50.2, 45.3, 41.3, 21675),
        "FlatQuant": (60.4, 81.6, 77.6, 79.6, 72.6, 66.9, 73.1, 11.4),
        "SVDQuant": (56.8, 80.4, 76.7, 78.4, 71.4, 64.9, 71.4, 12.8),
        "TwinQuant": (58.0, 82.1, 78.6, 79.0, 72.8, 66.1, 72.8, 11.8),
    },
    "qwen3_32b": {
        "W16A16": (57.8, 84.4, 84.2, 80.9, 73.6, 70.3, 75.2, 7.6),
        "RTN": (28.5, 27.6, 60.8, 52.5, 50.7, 52.0, 45.4, 1796),
        "SpinQuant": (56.7, 80.6, 79.2, 78.9, 71.8, 64.8, 72.0, 12.1),
        "QuaRot": (49.8, 72.2, 75.6, 73.4, 67.9, 65.3, 67.4, 15.6),
        "SmoothQuant": (26.0, 28.9, 56.7, 51.9, 54.1, 48.6, 44.4, 1806),
        "FlatQuant": (57.5, 81.2, 80.6, 80.2, 71.5, 65.9, 72.8, 11.0),
        "SVDQuant": (56.1, 80.9, 81.1, 78.6, 70.8, 65.6, 72.2, 11.6),
        "TwinQuant": (57.3, 82.4, 82.6, 79.8, 72.5, 65.2, 73.3, 10.4),
    },
}
TWINQUANT_METHODS = ("W16A16", "QuaRot", "SpinQuant", "FlatQuant", "SVDQuant", "TwinQuant")
OUR_ROWS = {"bf16": "bf16 (ours)", "hadamard_asym_g128": "Hadamard (ours)",
            "nar_k8_asym_g128": "PrismQuant k=8", "nar_kmax_asym_g128": "PrismQuant k=max"}
HARNESS_TASK = {"ARC-C": "arc_challenge", "ARC-E": "arc_easy", "HellaSwag": "hellaswag",
                "PIQA": "piqa", "Winogrande": "winogrande", "LAMBADA": "lambada_openai"}
ASSUMED = ("checkpoint (post-trained)", "context 2048 and BOS-per-window perplexity", "harness version",
           "acc_norm where defined", "KV cache unquantized", "all linear inputs quantized, attention matmuls not")


# Qwen3-32B in fp32 containers is 128 GB, so it is sharded across every
# visible GPU for calibration, control and evaluation (E21's loader and its
# per-device rotation factors, here combined with E19's Qwen3 R4 loading) and
# held on the CPU for the GPTQ layer loop, exactly as the 70B was.
SHARDED = {"qwen3_32b"}
CURRENT_COMMAND = ""


class ShardedQwen3RotationSet(e21.ShardedRotationSet, e19.Qwen3RotationSet):
    """E21's per-device factor routing over E19's Qwen3 rotation set."""


_load_model_fp32_single = e19.load_model_fp32


def _configure_sharded(model_key: str) -> None:
    e19.load_model_fp32 = _load_model_fp32_single
    if model_key not in SHARDED:
        return
    # E22 and E19 load the bf16 row and the control through e19.load_model_fp32
    # by name, so the sharded loader is installed there as well.
    e19.load_model_fp32 = lambda workdir: e21.load_model_sharded(e19.MODEL_ID, Path(workdir))
    if CURRENT_COMMAND == "gptq":
        e14.LOAD_MODEL = lambda model_id, workdir: e21.load_model_cpu(model_id, Path(workdir))
    else:
        e14.LOAD_MODEL = lambda model_id, workdir: e21.load_model_sharded(model_id, Path(workdir))
    e14.ROTATION_SET = ShardedQwen3RotationSet
    e14.FOLD_DEVICE = torch.device("cuda:0")


_configure_e22 = e22.configure


def _configure(model_key: str) -> None:
    _configure_e22(model_key)
    _configure_sharded(model_key)


# Two quantizer settings share TwinQuant's task set, checkpoint, context and
# 16-bit KV cache. "asym" is this repository's own quantizer (the one the
# Llama and E22 rows use: GPTQ g128_asym weights, asymmetric group-128
# activations) and is the bridge's headline; "sym" is TwinQuant's symmetric
# group-128 quantizer for both, kept as a diagnostic (its rows are e23sym_*).
QUANTIZERS = {
    "asym": {"prefix": "e23", "protocol": "g128_asym", "activation": "asymmetric_g128"},
    "sym": {"prefix": "e23sym", "protocol": "g128", "activation": "symmetric_g128"},
}


def configure_e23(quantizer: str = "asym") -> None:
    """Re-point E22 at TwinQuant's task protocol before its parser runs."""
    q = QUANTIZERS[quantizer]
    e22.configure = _configure
    e22.PREFIX = q["prefix"]
    e22.PROTOCOL = q["protocol"]
    e22.ACTIVATION_KIND = q["activation"]
    e22.QUANTIZE_KV = False
    e22.ARTIFACT_SUBDIR = "e23"
    e22.FAMILY = E23_FAMILY
    e22.DEFAULT_BATCH = {"qwen3_8b": 8, "qwen3_14b": 4, "qwen3_32b": 2}
    e22.DEFAULT_BENCHMARKS = BENCHMARKS
    e22.MATH_BENCHMARK = {}


def bridge_command(args: argparse.Namespace) -> None:
    """results/e23_bridge_summary.csv: TwinQuant's rows and ours, one line each."""
    rows: list[dict[str, Any]] = []
    for model in E23_FAMILY:
        published = TWINQUANT[model]
        for method in TWINQUANT_METHODS:
            v = published[method]
            rows.append({"model": model, "checkpoint": "unstated (Qwen3 " + model.split("_")[1].upper() + ")",
                         "source": "TwinQuant Table 5", "row": method, "context": "unstated",
                         "W/A/KV": "16/16/16" if method == "W16A16" else "4/4/16",
                         **{t: v[i] for i, t in enumerate(TASKS)}, "six_task_mean": v[6], "wikitext2_ppl": v[7],
                         "ppl_rel_degradation_pct": None if method == "W16A16" else round(100 * (v[7] / published["W16A16"][7] - 1), 1),
                         "acc_delta_vs_16bit": None if method == "W16A16" else round(v[6] - published["W16A16"][6], 2),
                         "assumed_settings": ""})
        directory = e22.WORKDIR / "results" / model
        for quantizer, q in QUANTIZERS.items():
          ours: dict[str, dict[str, Any]] = {}
          for row in OUR_ROWS:
            prefix = "e23" if row == "bf16" else q["prefix"]
            ppl = directory / f"{prefix}_{row}_wikitext.json"
            six = directory / f"{prefix}_{row}_six_task.json"
            entry: dict[str, Any] = {}
            if ppl.exists():
                entry["ppl"] = json.loads(ppl.read_text())["headline"]
            if six.exists():
                payload = json.loads(six.read_text())
                results = payload["results"]
                for col, task in HARNESS_TASK.items():
                    metric = "acc_norm,none" if "acc_norm,none" in results[task] else "acc,none"
                    entry[col] = round(100 * float(results[task][metric]), 2)
                entry["mean"] = payload["headline"]
            if entry:
                ours[row] = entry
          ref = ours.get("bf16", {})
          for row, label in OUR_ROWS.items():
            if row not in ours or (row == "bf16" and quantizer != "asym"):
                continue
            e = ours[row]
            wbits = "16/16/16" if row == "bf16" else ("4.156/4.25/16" if quantizer == "asym" else "4.125/4.125/16")
            rows.append({"model": model, "checkpoint": e22.act.MODEL_IDS[model],
                         "source": "this work (E23)" if quantizer == "asym" else "this work (E23, TwinQuant's symmetric quantizer)",
                         "row": label if quantizer == "asym" else f"{label}, sym", "context": 2048, "W/A/KV": wbits,
                         **{t: e.get(t) for t in TASKS}, "six_task_mean": e.get("mean"), "wikitext2_ppl": e.get("ppl"),
                         "ppl_rel_degradation_pct": (round(100 * (e["ppl"] / ref["ppl"] - 1), 1)
                                                     if row != "bf16" and "ppl" in e and "ppl" in ref else None),
                         "acc_delta_vs_16bit": (round(e["mean"] - ref["mean"], 2)
                                                if row != "bf16" and "mean" in e and "mean" in ref else None),
                         "assumed_settings": "; ".join(ASSUMED)})
    out = e22.WORKDIR / "results" / "e23_bridge_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    for r in rows:
        print(f"{r['model']:10s} {r['row']:18s} ppl={r['wikitext2_ppl']!s:>8} mean={r['six_task_mean']!s:>6} "
              f"rel={r['ppl_rel_degradation_pct']!s:>6} dacc={r['acc_delta_vs_16bit']!s:>6}")
    print(f"wrote {out} ({len(rows)} rows)")


def main() -> None:
    quantizer = "sym" if "--quantizer=sym" in sys.argv else "asym"
    sys.argv = [a for a in sys.argv if not a.startswith("--quantizer=")]
    configure_e23(quantizer)
    parser = e22.parser()
    parser.description = __doc__
    parser._subparsers._group_actions[0].add_parser("bridge")  # type: ignore[union-attr]
    args = parser.parse_args()
    global CURRENT_COMMAND
    CURRENT_COMMAND = args.command
    e22.WORKDIR = Path(args.workdir).resolve()
    args.artifact_root = str(e22.artifact_root())
    if args.command == "bridge":
        bridge_command(args)
        return
    {
        "audit": e22.audit_command, "calibrate": e22.calibrate_command, "control": e22.control_command,
        "gptq": e22.gptq_command, "evaluate": e22.evaluate_command, "gate": e22.gate_command,
        "benchmark-config": e22.benchmark_config_command, "finalize": e22.finalize_command,
    }[args.command](args)


if __name__ == "__main__":
    main()
