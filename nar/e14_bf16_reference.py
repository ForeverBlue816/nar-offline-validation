"""E14 16-bit reference row.

E14's table reports W4A4KV4 rows only, so the degradation relative to the
unquantized model was never measurable.  External perplexities cannot supply
that denominator: WikiText-2 perplexity depends on the chunking, and E14 scores
141 contiguous 2048-token windows with no BOS.  This module measures the
denominator under E14's own evaluation path.

Everything that touches the number is imported from e14 rather than
reimplemented: the token stream comes from ``_full_wikitext_tokens`` and the
per-chunk loss from ``_evaluate_ppl``, so the reference and the quantized rows
differ only by the rotation, the fake quantization and the KV hooks.  In
particular the model stays bf16 and the loss stays HuggingFace's, because E14's
rows are scored that way; E19 evaluates in fp32 containers instead, and mixing
the two paths would produce a denominator that does not subtract.

The plain checkpoint is loaded without norm fusion.  Fusion is exact in real
arithmetic but rounds in bf16, and the 16-bit reference is defined as the model
as published, which is also what the published baselines quote.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nar import activation_experiments as act  # noqa: E402
from nar import e14_w4a4kv4 as e14  # noqa: E402
from nar import experiment as base  # noqa: E402

LOG = base.LOG


def _tokens(model_id: str, workdir: Path, seq_len: int) -> torch.Tensor:
    """E14's token stream, preferring the file E14 itself wrote.

    ``model_key_from_id`` returned a hash for this checkpoint before it was
    added to MODEL_IDS, so the cache E14 populated sits under the hashed name.
    Recomputing gives the same tensor, but reading the original file removes
    any doubt that the reference scores the same tokens as the rows.
    """
    legacy = workdir / "cache" / "tokenized" / f"{hashlib.sha1(model_id.encode()).hexdigest()[:12]}-wikitext2-test-full-l{seq_len}.pt"
    if legacy.exists():
        LOG.info("using the token stream E14 wrote: %s", legacy)
        return torch.load(legacy, map_location="cpu", weights_only=True)
    return e14._full_wikitext_tokens(model_id, workdir, seq_len)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--model", default="llama31_8b", choices=tuple(act.MODEL_IDS))
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expect-chunks", type=int, default=0,
                        help="fail unless the token stream has this many windows")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the E14 reference row requires CUDA")
    workdir = Path(args.workdir).resolve()
    model_id, model_key = act.model_id_and_key(args.model)
    path = workdir / "results" / model_key / f"e14_bf16_reference_seed{args.seed}_ppl.json"
    if path.exists():
        LOG.info("E14 reference row exists: %s", path)
        return

    base.setup_logging(workdir, f"e14-bf16-reference-{model_key}")
    base.seed_everything(args.seed)
    tokens = _tokens(model_id, workdir, args.seq_len)
    if args.expect_chunks and tokens.shape[0] != args.expect_chunks:
        raise RuntimeError(
            f"token stream has {tokens.shape[0]} windows, expected {args.expect_chunks}; "
            "the reference would not subtract from the E14 rows"
        )
    model = base.load_model(model_id, workdir)
    ppl, chunk_rows = e14._evaluate_ppl(model, tokens, f"{model_key} bf16_reference")
    base.atomic_json(path, {
        "model": model_key, "model_id": model_id, "row": "bf16_reference",
        "rotation_checkpoint": None, "ppl": ppl, "chunks": chunk_rows,
        "dataset": "WikiText-2 raw test full contiguous token stream",
        "sequence_length": args.seq_len, "chunks_evaluated": len(chunk_rows),
        "weights": "bfloat16 as published, no norm fusion, no rotation, no quantization",
        "loss": "HuggingFace CausalLM loss, identical to the E14 quantized rows",
        "seed": args.seed, "hardware": base.hardware_info(),
    })
    LOG.info("E14 bf16 reference %s ppl=%.6f over %d windows", model_key, ppl, len(chunk_rows))


if __name__ == "__main__":
    main()
