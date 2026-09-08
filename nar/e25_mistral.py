"""E25 — Mistral-7B-v0.3 through the E22 pipeline, unchanged.

The third architecture family. Mistral-7B-v0.3 is Llama-shaped (GQA 32/8,
head_dim 128, no biases, RMSNorm, untied embeddings, vocabulary 32768, no
sliding window), so the E22 pipeline runs on it with two substitutions:
the architecture audit asserts Mistral's fields instead of Qwen3's
q_norm/k_norm, and the model key. Everything else — R1/R2/R4 calibration,
fp32 fold, exact-transpose control, GPTQ, the KIVI cache, the harness — is
E22's code.

Rows (seed 0), main protocol g128_asym (prefix ``e25``): bf16, Hadamard,
PrismQuant k=8, PrismQuant k=max, W4A4KV4. The bridge pair (prefix
``e25pc``): Hadamard and PrismQuant k=max under the default per-channel
GPTQ protocol, which is the weight quantizer the published Mistral W4A4
tables (QuaRot, DuQuant, FlatQuant, SpinQuant) use.

Benchmarks: WikiText-2 and C4 perplexity at 2048 (E14 chunking), the
eight-task zero-shot suite and the six-task mean.

  python nar/e25_mistral.py --workdir W --model mistral_7b_v03 [--protocol=default] <command>
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
from nar import e22_qwen3_family as e22  # noqa: E402
from nar import experiment as base  # noqa: E402

FAMILY = ("mistral_7b_v03", "mistral_7b_v01")
BENCHMARKS = ("wikitext", "c4", "eight_task", "six_task")
PROTOCOLS = {"g128_asym": "e25", "default": "e25pc"}
EXPECTED = {"num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32}


def mistral_architecture_audit(model: torch.nn.Module) -> dict[str, Any]:
    """Every field the spec names, asserted; no sliding-window attention."""
    config = model.config
    problems: list[str] = []
    shapes = {
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
    }
    for key, want in EXPECTED.items():
        if shapes[key] != want:
            problems.append(f"{key}={shapes[key]} expected {want}")
    architecture = config.architectures[0] if config.architectures else None
    if architecture != "MistralForCausalLM":
        problems.append(f"architecture {architecture}, expected MistralForCausalLM")
    sliding = getattr(config, "sliding_window", None)
    if sliding is not None:
        problems.append(f"sliding_window={sliding}; v0.3 must have none")
    layer_types = getattr(config, "layer_types", None)
    if layer_types and any(t != "full_attention" for t in layer_types):
        problems.append(f"layer_types contain non-full attention: {sorted(set(layer_types))}")
    if int(config.vocab_size) != 32768:
        problems.append(f"vocab_size={config.vocab_size}, expected 32768 (v0.3)")
    head = model.get_output_embeddings()
    tied = bool(getattr(config, "tie_word_embeddings", False))
    shared = bool(head is not None and head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())
    if tied or shared:
        problems.append(f"tie_word_embeddings={tied} shared_storage={shared}; expected untied")
    biased = [name for name, module in model.named_modules()
              if isinstance(module, torch.nn.Linear) and module.bias is not None]
    if biased:
        problems.append(f"biased linear modules present: {biased[:5]}")
    norm_kinds = sorted({module.__class__.__name__ for module in model.modules()
                         if module.__class__.__name__.endswith("Norm")})
    if norm_kinds != ["MistralRMSNorm"]:
        problems.append(f"norm kinds {norm_kinds}, expected only MistralRMSNorm")
    fused_norms = sorted({name.split(".")[-1] for name, module in model.named_modules()
                          if module.__class__.__name__.endswith("RMSNorm")}
                         & {"input_layernorm", "post_attention_layernorm", "norm"})
    qk_norms = [name for name, _ in model.named_modules() if name.endswith(("q_norm", "k_norm"))]
    if qk_norms:
        problems.append(f"unexpected q_norm/k_norm modules: {qk_norms[:3]}")
    if e19.MODEL_KEY == "mistral_7b_v01":
        problems = [f"(anchor model, not asserted) {p}" for p in problems]
    audit = {
        "model_id": e19.MODEL_ID, "architecture": architecture, "shapes": shapes,
        "vocab_size": int(config.vocab_size), "sliding_window": sliding,
        "rope_theta": getattr(config, "rope_theta", None), "rms_norm_eps": getattr(config, "rms_norm_eps", None),
        "hidden_act": getattr(config, "hidden_act", None),
        "slot_counts": {"qkv": shapes["hidden_size"] // e14.GROUP, "down": shapes["intermediate_size"] // e14.GROUP},
        "kv_group_counts": {"k_per_channel_groups": shapes["head_dim"] // e14.K_TOKEN_GROUP,
                            "v_per_token_group_size": shapes["head_dim"], "kv_heads": shapes["num_key_value_heads"],
                            "num_key_value_groups": shapes["num_attention_heads"] // shapes["num_key_value_heads"]},
        "tie_word_embeddings": tied, "embedding_and_lm_head_share_storage": shared,
        "linear_modules_with_bias": biased, "rmsnorm_kinds_fused_into_consumers": fused_norms,
        "per_head_qk_norm_modules": 0,
        "hadamard_orders": {"hidden 4096": "2^12", "intermediate 14336": "28 x 512 (Paley 28, as Llama-3.1-8B)", "heads 32": "2^5"},
        "problems": problems,
    }
    if problems and e19.MODEL_KEY != "mistral_7b_v01":
        raise AssertionError("Mistral architecture audit failed:\n  " + "\n  ".join(problems))
    return audit


def configure_e25(protocol: str) -> None:
    e22.PREFIX = PROTOCOLS[protocol]
    e22.PROTOCOL = protocol
    e22.ACTIVATION_KIND = "asymmetric_g128"
    e22.QUANTIZE_KV = True
    e22.ARTIFACT_SUBDIR = "e25"
    e22.FAMILY = FAMILY
    e22.DEFAULT_BATCH = {"mistral_7b_v03": 8, "mistral_7b_v01": 8}
    e22.DEFAULT_BENCHMARKS = BENCHMARKS
    e22.MATH_BENCHMARK = {}
    e22.TECH_REPORT = {}
    e19.EXPECTED = dict(EXPECTED)
    e19.architecture_audit = mistral_architecture_audit
    # E19's probe hooks Qwen3's k_norm; Mistral has none, and K is quantized
    # post-RoPE inside the registered attention function as on Llama (E14).
    e19.kv_site_probe = lambda model: {
        "note": "no q_norm/k_norm on Mistral; K and V are taken inside the registered attention "
                "function after RoPE (E14 Llama path), so the KV quantizer sees the same tensors "
                "as on Llama-3.x"}
    # The rotation-only control is protocol-independent; both prefixes read
    # the one written under e25.
    e19.control_path = lambda workdir: Path(workdir) / "results" / e19.MODEL_KEY / "e25_rotation_only_control.csv"


def summary_command(args: argparse.Namespace) -> None:
    """results/mistral_7b_v03/e25_summary.csv and E25_DONE.json."""
    directory = e22.WORKDIR / "results" / "mistral_7b_v03"
    rows: list[dict[str, Any]] = []
    for protocol, prefix in PROTOCOLS.items():
        for path in sorted(directory.glob(f"{prefix}_*_*.json")):
            payload = json.loads(path.read_text())
            if "headline" not in payload or "row" not in payload:
                continue
            rows.append({"protocol": protocol, "row": payload["row"], "benchmark": payload["benchmark"],
                         "headline_metric": payload["headline_metric"], "value": payload["headline"],
                         "effective_bits": json.dumps(payload.get("effective_bits")),
                         "kv_policy": payload.get("kv_policy"), "git_commit": payload.get("git_commit"),
                         "artifact": str(path.relative_to(e22.WORKDIR))})
    base.write_csv(directory / "e25_summary.csv", rows)
    complete = {(r["protocol"], r["row"], r["benchmark"]) for r in rows}
    wanted = {("g128_asym", row, bench) for row in e22.ROWS if not row.startswith("w_only_") for bench in BENCHMARKS}
    wanted |= {("default", row, bench) for row in ("hadamard_asym_g128", "nar_kmax_asym_g128") for bench in BENCHMARKS}
    missing = sorted(wanted - complete)
    done = {"model": "mistral_7b_v03", "model_id": e19.MODEL_ID, "rows": len(rows), "missing": missing,
            "complete": not missing, "git_commit": e19.git_commit()}
    base.atomic_json(directory / "E25_DONE.json", done)
    for r in rows:
        print(f"{r['protocol']:10s} {r['row']:22s} {r['benchmark']:10s} {r['value']:9.3f}")
    print(f"missing: {missing}")


def main() -> None:
    protocol = "default" if "--protocol=default" in sys.argv else "g128_asym"
    sys.argv = [a for a in sys.argv if not a.startswith("--protocol=")]
    configure_e25(protocol)
    parser = e22.parser()
    parser.description = __doc__
    parser._subparsers._group_actions[0].add_parser("summary")  # type: ignore[union-attr]
    args = parser.parse_args()
    e22.WORKDIR = Path(args.workdir).resolve()
    args.artifact_root = str(e22.artifact_root())
    if args.command == "summary":
        summary_command(args)
        return
    {
        "audit": e22.audit_command, "calibrate": e22.calibrate_command, "control": e22.control_command,
        "gptq": e22.gptq_command, "evaluate": e22.evaluate_command, "gate": e22.gate_command,
        "benchmark-config": e22.benchmark_config_command, "finalize": e22.finalize_command,
    }[args.command](args)


if __name__ == "__main__":
    main()
