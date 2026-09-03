#!/usr/bin/env python3
"""E16 post-hoc SmoothQuant robustness in the frozen E11 setting.

This is deliberately a one-seed diagnostic.  The metadata-matched Hadamard and
NAR reference rows are reused from E11; only the three requested SmoothQuant
variants are evaluated here.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from . import e11_fair_baselines as e11
except ImportError:
    import activation_experiments as act
    import experiment as base
    import e11_fair_baselines as e11


VARIANTS = {
    "smoothquant_a065_both": (0.65, False),
    "smoothquant_a080_both": (0.80, False),
    "smoothquant_a050_qkv_only": (0.50, True),
}


class RobustTransform(e11.Transform):
    """E11 SmoothQuant+Hadamard with a frozen alpha/site placement."""

    def __init__(self, *args: Any, qkv_only: bool, **kwargs: Any) -> None:
        self.qkv_only = qkv_only
        super().__init__(*args, **kwargs)

    def activation(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        x = value.float()
        if not self.qkv_only or site == "qkv":
            scale = self.stats["smoothquant_scale"][e11._key(site, layer)].to(self.device)
            x = x / scale
        return self._orthogonal(site, layer, x)

    def weight(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        w = value.float()
        if not self.qkv_only or site == "qkv":
            scale = self.stats["smoothquant_scale"][e11._key(site, layer)].to(self.device)
            w = w * scale
        return self._orthogonal(site, layer, w)


def _stats_for_alpha(source: dict[str, Any], alpha: float) -> dict[str, Any]:
    """Recover weight maxima from the frozen alpha=.5 scale, then change alpha."""
    activation = source["activation_absmax"]
    old_scale = source["smoothquant_scale"]
    scales: dict[str, torch.Tensor] = {}
    for key, act_max in activation.items():
        a = act_max.float().clamp_min(1e-8)
        # At alpha=.5, s^2=a/w.  This avoids another calibration/model pass.
        w = (a / old_scale[key].float().clamp_min(1e-8).square()).clamp_min(1e-8)
        scales[key] = a.pow(alpha) / w.pow(1.0 - alpha)
    return {"activation_absmax": activation, "smoothquant_scale": scales}


def _mean_ppl(rows: list[dict[str, Any]]) -> float:
    return math.exp(sum(float(row["nll"]) for row in rows) / len(rows))


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("E16 requires an allocated CUDA GPU")
    workdir = Path(args.workdir).resolve()
    asset_workdir = Path(args.asset_workdir).resolve() if args.asset_workdir else workdir
    model_id, model_key = act.model_id_and_key(args.model)
    base.setup_logging(workdir, f"e16-smoothquant-{model_key}")
    result_dir = workdir / "results" / model_key
    result_dir.mkdir(parents=True, exist_ok=True)
    done = result_dir / "E16_SMOOTHQUANT_DONE.json"
    if done.exists():
        return

    calibration = e11.calibration_dir(workdir, model_key)
    meta = json.loads((calibration / "DONE.json").read_text())
    frozen_stats = torch.load(
        calibration / "channel_stats.pt", map_location="cpu", weights_only=True
    )
    tokens = base.prepare_token_chunks(
        model_id, "test", 0, args.eval_sequences, args.seq_len, asset_workdir
    )
    model = base.load_model(model_id, asset_workdir)
    layers = int(model.config.num_hidden_layers)
    if layers != int(meta["layers"]):
        raise RuntimeError("E11/E16 layer mismatch")
    dimensions = {
        "qkv": int(model.config.hidden_size),
        "down": int(model.config.intermediate_size),
    }
    seed = args.seed
    partial = result_dir / "e16_smoothquant_per_sequence.partial.csv"
    rows: list[dict[str, Any]] = list(base.read_csv(partial)) if partial.exists() else []
    completed = {str(row["method"]) for row in rows}
    manager = e11.WeightManager(model, layers)

    for method, (alpha, qkv_only) in VARIANTS.items():
        if method in completed:
            continue
        stats = _stats_for_alpha(frozen_stats, alpha)
        transform = RobustTransform(
            "smoothquant_hadamard_g128_asym", model_key, workdir,
            0, seed, layers, dimensions, torch.device("cuda"), stats,
            qkv_only=qkv_only,
        )
        manager.restore()
        fold_error = manager.apply(transform, args.weight_row_batch)
        hooks = e11.Hooks(model, transform)
        hooks.install()
        try:
            with torch.inference_mode():
                losses = act.evaluate_nlls(model, tokens, f"{model_key} E16 {method}")
        finally:
            hooks.close()
        rows.extend({
            "model": model_key,
            "method": method,
            "seed": seed,
            "sequence": index,
            "nll": loss,
            "tokens_scored": args.seq_len - 1,
            "alpha": alpha,
            "smoothing_sites": "qkv" if qkv_only else "qkv+down",
            "weight_fold_max_relative_error": fold_error,
        } for index, loss in enumerate(losses))
        base.write_csv(partial, rows)
        del transform, stats
        gc.collect()
        torch.cuda.empty_cache()

    manager.restore()
    references = [
        row for row in base.read_csv(result_dir / "e11_per_sequence.csv")
        if int(row["seed"]) == seed
        and row["method"] in ("hadamard_g128_asym", "nar_b128_kmax")
    ]
    reference_ppl = {
        method: _mean_ppl([row for row in references if row["method"] == method])
        for method in ("hadamard_g128_asym", "nar_b128_kmax")
    }
    summaries = []
    for method, (alpha, qkv_only) in VARIANTS.items():
        subset = [row for row in rows if row["method"] == method]
        ppl = _mean_ppl(subset)
        summaries.append({
            "model": model_key,
            "method": method,
            "seed": seed,
            "alpha": alpha,
            "smoothing_sites": "qkv" if qkv_only else "qkv+down",
            "ppl": ppl,
            "ppl_delta_vs_hadamard": ppl - reference_ppl["hadamard_g128_asym"],
            "ppl_delta_vs_nar_kmax": ppl - reference_ppl["nar_b128_kmax"],
            "effective_activation_bits": 4.25,
            "paired_ci": "not estimable with one seed",
        })
    base.write_csv(result_dir / "e16_smoothquant_per_sequence.csv", rows)
    base.write_csv(result_dir / "e16_smoothquant_summary.csv", summaries)
    base.atomic_json(done, {
        "model": model_key,
        "model_id": model_id,
        "post_hoc": True,
        "seed": seed,
        "seed_count": 1,
        "eval_sequences": args.eval_sequences,
        "sequence_length": args.seq_len,
        "variants": VARIANTS,
        "references_reused_without_rerun": ["E11 hadamard_g128_asym", "E11 nar_b128_kmax"],
        "scale_formula": "s=max|x|^alpha/max|w|^(1-alpha); x'=x/s; W'=W*s",
        "weight_max_recovery": "from frozen E11 alpha=.5 statistics: max|w|=max|x|/s_.5^2",
        "effective_bits": "4 + (16-bit scale + 16-bit zero-point)/128 = 4.25",
        "paired": "same 64 WikiText-2 test chunks and same seed as reused E11 rows",
        "paired_ci": "not estimable with one seed",
        "no_tuning": True,
        "negative_results_reported": True,
        "summary": summaries,
        "hardware": base.hardware_info(),
    })
    partial.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--asset-workdir")
    result.add_argument("--model", choices=e11.MODELS, required=True)
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--eval-sequences", type=int, default=64)
    result.add_argument("--seq-len", type=int, default=2048)
    result.add_argument("--weight-row-batch", type=int, default=512)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
