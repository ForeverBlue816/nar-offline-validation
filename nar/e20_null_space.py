#!/usr/bin/env python3
"""E20 - does the quantizer's free direction generalize beyond the zero-point?

Activation-only, in the E11 setting, with two requirements carried over from
E18 v2: the fold is the exact transpose ``x -> R^T Q(R x)`` so no rotated row
carries weight rounding the reference lacks, and per-chunk NLL is fp32.  No row
is copied from E11; every row is re-run here so all deltas are paired.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from . import activation_experiments as act
    from . import experiment as base
    from . import quantizer_affine as aq
    from .e12_wy import compact_wy
    from .e18_v2 import full_hadamard_rows_transpose
except ImportError:
    import activation_experiments as act
    import experiment as base
    import quantizer_affine as aq
    from e12_wy import compact_wy
    from e18_v2 import full_hadamard_rows_transpose


LOG = logging.getLogger("nar")
MODELS = ("llama32_3b", "llama31_8b")
SITES = ("qkv", "down")
SEEDS = (0, 1, 2)
EVAL_CHUNKS = 64


@dataclass(frozen=True)
class Row:
    method: str          # "bf16" | "hadamard" | "nar"
    group: int
    m: int

    @property
    def name(self) -> str:
        if self.method == "bf16":
            return "bf16"
        return f"{self.method}_g{self.group}_m{self.m}"

    @property
    def bits(self) -> float:
        return 16.0 if self.method == "bf16" else aq.effective_bits(self.group, self.m)

    def slots(self, n: int) -> int:
        return 0 if self.method != "nar" else (n // self.group) * self.m


ROWS = (
    Row("bf16", 0, 0),
    Row("hadamard", 64, 1), Row("hadamard", 128, 1), Row("hadamard", 256, 1),
    Row("hadamard", 256, 2), Row("hadamard", 256, 3),
    Row("nar", 64, 1), Row("nar", 128, 1), Row("nar", 256, 1),
    Row("nar", 256, 2), Row("nar", 256, 3), Row("nar", 128, 2),
)
BASELINE = "nar_g128_m1"


# ------------------------------------------------------- alignment target ---

def anchor_offsets(group: int, m: int) -> list[int]:
    """Within-group coordinates whose Hadamard image is w_1..w_m.

    The normalized Hadamard is symmetric and involutive, so ``H e_c`` is Walsh
    row ``c`` in the FWHT's natural (Sylvester) order.  The sequency rows the
    protocol names sit at 0, g/2 and 3g/4 there.  The mapping is asserted
    numerically rather than trusted, because it is the crux of the whole
    construction.
    """
    offsets = [0, group // 2, group // 2 + group // 4][:m]
    if len(offsets) != m:
        raise ValueError(f"m={m} exceeds the three named Walsh directions")
    basis = aq.walsh_basis(group, m)
    identity = torch.eye(group)
    imaged = act.ext._fast_walsh_hadamard(identity[offsets])
    for j, offset in enumerate(offsets):
        if not torch.allclose(imaged[j].abs(), basis[j].abs(), atol=1e-5):
            raise AssertionError(f"H e_{offset} is not Walsh row {j} for g={group}")
    return offsets


def null_space_basis(n: int, group: int, m: int, device: torch.device) -> torch.Tensor:
    """P_N as an (G*m, n) orthonormal matrix: w_j embedded in each group."""
    basis = aq.walsh_basis(group, m, device)
    groups = n // group
    out = torch.zeros((groups * m, n), device=device)
    for g in range(groups):
        for j in range(m):
            out[g * m + j, g * group:(g + 1) * group] = basis[j]
    gram = out @ out.T
    error = float((gram - torch.eye(out.shape[0], device=device)).abs().max())
    if error > 1e-4:
        raise AssertionError(f"embedded P_N basis not orthonormal: {error}")
    return out


def reflectors_to_anchors(vectors: torch.Tensor, anchors: list[int]
                          ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Householders mapping column i to coordinate ``anchors[i]``.

    Generalizes ``ext._householders_to_anchors``, which hardcodes ``i * b`` and
    therefore cannot express more than one slot per group.
    """
    work = vectors.float().clone()
    n, rank = work.shape
    reflectors: list[torch.Tensor] = []
    active: list[bool] = []
    for index in range(rank):
        target = torch.zeros(n, dtype=work.dtype, device=work.device)
        target[anchors[index]] = 1.0
        delta = work[:, index] - target
        norm = delta.norm()
        if float(norm) < 1e-7:
            reflectors.append(torch.zeros(n, dtype=work.dtype, device=work.device))
            active.append(False)
            continue
        reflector = delta / norm
        work[:, index:] -= 2 * reflector[:, None] * (reflector @ work[:, index:])[None, :]
        reflectors.append(reflector)
        active.append(True)
    index_tensor = torch.tensor(anchors, device=work.device)
    columns = torch.arange(rank, device=work.device)
    mapped = work[index_tensor, columns]
    off = work.clone()
    off[index_tensor, columns] = 0
    error = max(float((mapped - 1).abs().max()), float(off.abs().max()))
    return (torch.stack(reflectors),
            torch.tensor(active, dtype=torch.bool, device=work.device), error)


def build_nar_factor(vectors: torch.Tensor, energy: torch.Tensor, group: int, m: int
                     ) -> act.RotationFactor:
    """NAR factor whose alignment target is P_N instead of P_DC.

    G maps the top ``rank`` directions onto the first ``rank`` coordinates, the
    permutation carries those onto the ``G*m`` Walsh slots, and the remaining
    coordinates are distributed by the unchanged energy-balanced rule.
    """
    n = vectors.shape[0]
    groups = n // group
    offsets = anchor_offsets(group, m)
    rank = min(vectors.shape[1], groups * m)
    reflectors, active, error = reflectors_to_anchors(vectors[:, :rank], list(range(rank)))

    slots = [g * group + offsets[j] for g in range(groups) for j in range(m)][:rank]
    absorbed = list(range(rank))
    absorbed_set = set(absorbed)
    remaining = [index for index in range(n) if index not in absorbed_set]
    scores = energy.detach().float().cpu()
    # Low-energy fillers take the leftover Walsh slots, exactly as in E11.
    leftover = [g * group + offsets[j] for g in range(groups) for j in range(m)][rank:]
    fillers = sorted(remaining, key=lambda i: (float(scores[i]), i))[: len(leftover)]
    filler_set = set(fillers)
    residual = [index for index in remaining if index not in filler_set]
    residual.sort(key=lambda i: (-float(scores[i]), i))
    taken = set(slots) | set(leftover)
    free_slots = [p for p in range(n) if p not in taken]
    source = absorbed + fillers + residual
    target = slots + leftover + free_slots
    if sorted(source) != list(range(n)) or sorted(target) != list(range(n)):
        raise AssertionError("E20 permutation is not a bijection")
    device = vectors.device
    return act.RotationFactor(
        n=n, b=group, reflectors=reflectors, active=active,
        source_order=torch.tensor(source, device=device),
        target_order=torch.tensor(target, device=device),
        anchor_error=error,
    )


# ------------------------------------------------------ exact-transpose fold ---

class Rotation:
    """R and R^T for one row, with the round-trip residual recorded."""

    def __init__(self, row: Row, factors: dict[tuple[str, int], act.RotationFactor],
                 signs: dict[tuple[str, int], torch.Tensor]):
        self.row = row
        self.factors = factors
        self.signs = signs
        self.wy: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for key, factor in factors.items():
            self.wy[key] = compact_wy(factor.reflectors, factor.active)

    def apply(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        signs = self.signs[(site, layer)]
        if self.row.method == "hadamard":
            return act.full_hadamard_rows(value.float(), signs)
        return self.factors[(site, layer)].apply(value, signs)

    def apply_transpose(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        signs = self.signs[(site, layer)]
        if self.row.method == "hadamard":
            return full_hadamard_rows_transpose(value.float(), signs)
        factor = self.factors[(site, layer)]
        w, y = self.wy[(site, layer)]
        n = factor.n
        shape = value.shape
        rows = value.float().reshape(-1, n)
        rows = act.ext._fast_walsh_hadamard(
            rows.reshape(-1, n // factor.b, factor.b)).reshape(-1, n)
        rows = rows * signs
        unpermuted = torch.empty_like(rows)
        unpermuted[:, factor.source_order] = rows[:, factor.target_order]
        return (unpermuted - (unpermuted @ y) @ w.T).reshape(shape)

    def round_trip(self, site: str, layer: int, n: int, device: torch.device,
                   rows: int = 8) -> float:
        probe = torch.randn((rows, n), device=device, dtype=torch.float32)
        restored = self.apply_transpose(site, layer, self.apply(site, layer, probe))
        return float((restored - probe).norm() / probe.norm())


class Hooks:
    def __init__(self, model: torch.nn.Module, rotation: Rotation | None, row: Row, layers: int):
        self.model = model
        self.rotation = rotation
        self.row = row
        self.layers = layers
        self.handles: list[Any] = []

    def quantize(self, site: str, layer: int, value: torch.Tensor) -> torch.Tensor:
        if self.rotation is None:
            return value
        rotated = self.rotation.apply(site, layer, value)
        result = aq.quantize_affine(rotated, self.row.group, self.row.m).dequant
        return self.rotation.apply_transpose(site, layer, result).to(value.dtype)

    def install(self) -> None:
        for layer, block in enumerate(self.model.model.layers[: self.layers]):
            self.handles.append(block.input_layernorm.register_forward_hook(
                lambda _m, _i, output, layer=layer: self.quantize("qkv", layer, output)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(
                lambda _m, inputs, layer=layer: (self.quantize("down", layer, inputs[0]),)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def evaluate_ppl_fp32(model: torch.nn.Module, tokens: torch.Tensor, label: str
                      ) -> tuple[float, list[float]]:
    values: list[float] = []
    for index in range(tokens.shape[0]):
        batch = tokens[index:index + 1].cuda()
        logits = model(input_ids=batch, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        if loss.dtype != torch.float32:
            raise AssertionError(f"per-chunk NLL must be fp32, got {loss.dtype}")
        value = float(loss)
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite E20 loss at chunk {index}")
        values.append(value)
        if index % 16 == 0:
            LOG.info("%s chunk %d/%d nll=%.6f", label, index + 1, tokens.shape[0], value)
        del logits, loss
    return math.exp(float(np.mean(values))), values


# ----------------------------------------------------- eigenvector recovery ---

def e11_root(workdir: Path, model_key: str) -> Path:
    return workdir / "activations" / model_key / "e11_calibration"


def recover_eigenvectors(factor: act.RotationFactor) -> torch.Tensor:
    """The top directions E11 aligned, recovered exactly from its Householders.

    E11's G maps eigenvector i onto coordinate ``i * b``, so ``v_i = G^T e_{i*b}``
    and G^T is the same reflectors applied in reverse order.  This avoids a
    second calibration pass and guarantees E20 aligns the identical directions.
    """
    n, b = factor.n, factor.b
    rank = int(factor.reflectors.shape[0])
    device = factor.reflectors.device
    columns = torch.zeros((rank, n), device=device)
    columns[torch.arange(rank, device=device), torch.arange(rank, device=device) * b] = 1.0
    for index in range(rank - 1, -1, -1):
        if not bool(factor.active[index]):
            continue
        reflector = factor.reflectors[index]
        columns -= 2 * (columns @ reflector).unsqueeze(1) * reflector.unsqueeze(0)
    vectors = columns.T.contiguous()
    gram = vectors.T @ vectors
    error = float((gram - torch.eye(rank, device=device)).abs().max())
    if error > 1e-3:
        raise AssertionError(f"recovered eigenvectors are not orthonormal: {error}")
    return vectors


def load_site_data(workdir: Path, model_key: str, site: str, layer: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Top directions and the per-coordinate energy E11 used for its permutation."""
    path = e11_root(workdir, model_key) / "factors" / "nar_b64_kmax" / f"{site}_layer_{layer:02d}.pt"
    factor = act.RotationFactor.load(path, device)
    vectors = recover_eigenvectors(factor)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    energy = payload.get("permutation_energy")
    if energy is None:
        # E11 stored only the realized orders; reconstruct a monotone score
        # from them so the filler/residual rule is reproduced deterministically.
        order = factor.source_order.detach().cpu().float()
        energy = torch.empty_like(order)
        energy[order.long()] = torch.linspace(1.0, 0.0, order.numel())
    return vectors, energy.to(device)


def build_rotation(workdir: Path, model_key: str, row: Row, dimensions: dict[str, int],
                   layers: int, seed_index: int, base_seed: int,
                   device: torch.device) -> Rotation:
    factors: dict[tuple[str, int], act.RotationFactor] = {}
    signs: dict[tuple[str, int], torch.Tensor] = {}
    for site, n in dimensions.items():
        for layer in range(layers):
            generator = torch.Generator(device="cpu").manual_seed(
                base_seed + seed_index + 1000 * layer + (0 if site == "qkv" else 100_000))
            signs[(site, layer)] = torch.randint(
                0, 2, (n,), generator=generator).float().mul_(2).sub_(1).to(device)
            if row.method == "nar":
                vectors, energy = load_site_data(workdir, model_key, site, layer, device)
                factors[(site, layer)] = build_nar_factor(vectors, energy, row.group, row.m)
    return Rotation(row, factors, signs)


@torch.inference_mode()
def capture_site_samples(model: torch.nn.Module, tokens: torch.Tensor, layers: int,
                         device: torch.device, rows: int = 64) -> dict[str, torch.Tensor]:
    """Real activations at the two quantized sites; the c_j precision check
    must see the heavy-tailed data the quantizer actually meets."""
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def store(site: str) -> Callable[..., None]:
        def hook(_module, inputs, output=None):
            value = output if output is not None else inputs[0]
            if site not in captured:
                captured[site] = value.detach().reshape(-1, value.shape[-1])[:rows].float().to(device)
        return hook

    block = model.model.layers[0]
    handles.append(block.input_layernorm.register_forward_hook(
        lambda m, i, o: store("qkv")(m, i, o)))
    handles.append(block.mlp.down_proj.register_forward_pre_hook(
        lambda m, i: store("down")(m, i)))
    try:
        model(input_ids=tokens[:1].cuda(), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


# --------------------------------------------------------------- diagnostics ---

def eigen_fractions(workdir: Path, model_key: str) -> dict[tuple[str, int], list[float]]:
    """Per-direction share of the total activation second moment, from E11."""
    path = workdir / "results" / model_key / "e11_calibration_eigenspace.csv"
    table: dict[tuple[str, int], dict[int, float]] = {}
    for r in base.read_csv(path):
        table.setdefault((str(r["site"]), int(r["layer"])), {})[int(r["rank"])] = float(
            r["fraction_total_energy"])
    return {key: [value[rank] for rank in sorted(value)] for key, value in table.items()}


@torch.inference_mode()
def alignment_and_theory(workdir: Path, model_key: str, row: Row, rotation: Rotation,
                         dimensions: dict[str, int], layers: int, device: torch.device,
                         top: int = 96) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """s_i = ||P_N R v_i||^2 / ||v_i||^2, and f as the energy-weighted capture.

    f must weight each direction by its share of the total second moment, not
    count directions: an unweighted mean would report the fraction of the top
    directions that happen to receive a slot, which is a property of the slot
    budget rather than of the activation statistics.
    """
    align_rows: list[dict[str, Any]] = []
    theory_rows: list[dict[str, Any]] = []
    fractions = eigen_fractions(workdir, model_key)
    for site, n in dimensions.items():
        projector = null_space_basis(n, row.group, row.m, device)
        captured: list[float] = []
        for layer in range(layers):
            vectors, _ = load_site_data(workdir, model_key, site, layer, device)
            take = min(top, vectors.shape[1])
            rotated = rotation.apply(site, layer, vectors[:, :take].T)
            share = (rotated @ projector.T).square().sum(-1) / rotated.square().sum(-1).clamp_min(1e-30)
            weights = torch.tensor(fractions[(site, layer)][:take], device=device,
                                   dtype=torch.float32)
            captured.append(float((share[:weights.numel()] * weights).sum()))
            for index in range(take):
                align_rows.append({
                    "model": model_key, "row": row.name, "site": site, "layer": layer,
                    "direction": index, "captured_share": float(share[index]),
                    "energy_fraction": float(weights[index]) if index < weights.numel() else math.nan,
                })
        mean = float(np.mean(captured))
        theory_rows.append({
            "model": model_key, "row": row.name, "site": site,
            "group": row.group, "m": row.m, "slots": row.slots(n),
            "effective_bits": row.bits, "mean_captured_fraction": mean,
            "min_captured_fraction": float(np.min(captured)),
            "max_captured_fraction": float(np.max(captured)),
            "predicted_range_reduction": 1.0 - math.sqrt(max(0.0, 1.0 - mean)),
        })
        del projector
    return align_rows, theory_rows


@torch.inference_mode()
def coefficient_precision(rotation: Rotation, samples: dict[str, torch.Tensor],
                          row: Row, model_key: str) -> list[dict[str, Any]]:
    """Dequant error from storing c_j in fp16, relative to one quantization step."""
    out: list[dict[str, Any]] = []
    if row.m < 2:
        return out
    for site, sample in samples.items():
        rotated = rotation.apply(site, 0, sample)
        fp16 = aq.quantize_affine(rotated, row.group, row.m, torch.float16)
        fp32 = aq.quantize_affine(rotated, row.group, row.m, torch.float32)
        step = fp16.scale.float().reshape(-1)
        error = (fp16.dequant - fp32.dequant).abs().reshape(step.numel(), -1).amax(-1)
        ratio = (error / step.clamp_min(1e-30))
        # A ratio of exactly one step means a code flipped by one, so the code
        # flip rate is the interpretable quantity; the max ratio saturates.
        flips = (fp16.codes.to(torch.int16) != fp32.codes.to(torch.int16))
        out.append({
            "model": model_key, "row": row.name, "site": site,
            "max_abs_coefficient": float(fp32.coefficients.abs().max()),
            "max_dequant_error_from_fp16_c": float(error.max()),
            "mean_error_over_step": float(ratio.mean()),
            "max_error_over_step": float(ratio.max()),
            "code_flip_fraction": float(flips.float().mean()),
            "groups_with_a_flip": float(flips.any(-1).float().mean()),
            "exceeds_tenth_of_step": bool(float(ratio.max()) > 0.1),
        })
    return out


# ---------------------------------------------------------------- execution ---

TCRIT_DF2_90 = 2.919985580353725


def paired_ci(deltas: list[float]) -> tuple[float, float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    mean = float(values.mean())
    if values.size < 2:
        return mean, math.nan, math.nan
    half = TCRIT_DF2_90 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean, mean - half, mean + half


def per_sequence_path(workdir: Path, model_key: str) -> Path:
    return workdir / "results" / model_key / "e20_per_sequence.csv"


def run_model(args: argparse.Namespace, model_key: str) -> None:
    workdir = Path(args.workdir).resolve()
    model_id, _ = act.model_id_and_key(model_key)
    device = torch.device("cuda")
    tokens = base.prepare_token_chunks(model_id, "test", 0, args.chunks, args.seq_len, workdir)
    model = base.load_model(model_id, workdir)
    layers = len(model.model.layers)
    dimensions = {"qkv": int(model.config.hidden_size),
                  "down": int(model.config.intermediate_size)}
    result_dir = workdir / "results" / model_key
    path = per_sequence_path(workdir, model_key)
    rows: list[dict[str, Any]] = base.read_csv(path) if path.exists() else []
    done = {(str(r["row"]), int(r["seed"])) for r in rows}
    round_trips: list[dict[str, Any]] = []
    align_rows: list[dict[str, Any]] = []
    theory_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    samples: dict[str, torch.Tensor] = {}

    for row in ROWS:
        if row.method == "nar" and row.name not in args.rows and args.rows != ["all"]:
            continue
        if args.rows != ["all"] and row.name not in args.rows:
            continue
        seeds = (0,) if row.method == "bf16" else tuple(args.seeds)
        for seed_index in seeds:
            if (row.name, seed_index) in done:
                LOG.info("E20 row already complete: %s seed %d", row.name, seed_index)
                continue
            rotation = None
            if row.method != "bf16":
                rotation = build_rotation(workdir, model_key, row, dimensions, layers,
                                          seed_index, args.seed, device)
                for site, n in dimensions.items():
                    residual = rotation.round_trip(site, 0, n, device)
                    round_trips.append({"model": model_key, "row": row.name,
                                        "seed": seed_index, "site": site,
                                        "round_trip_relative_error": residual,
                                        "tolerance": args.round_trip_tolerance})
                    if residual > args.round_trip_tolerance:
                        raise AssertionError(f"round-trip check failed: {round_trips[-1]}")
                if seed_index == args.seeds[0]:
                    if not samples:
                        samples = capture_site_samples(model, tokens, layers, device)
                    precision_rows.extend(coefficient_precision(rotation, samples, row, model_key))
                    if row.method == "nar" or row.m > 1:
                        a, t = alignment_and_theory(workdir, model_key, row, rotation,
                                                    dimensions, layers, device)
                        align_rows.extend(a)
                        theory_rows.extend(t)
            hooks = Hooks(model, rotation, row, layers)
            hooks.install()
            try:
                ppl, values = evaluate_ppl_fp32(model, tokens, f"{model_key} {row.name} s{seed_index}")
            finally:
                hooks.close()
                del rotation
                gc.collect()
                torch.cuda.empty_cache()
            rows.extend({"model": model_key, "row": row.name, "method": row.method,
                         "group": row.group, "m": row.m, "seed": seed_index,
                         "chunk": index, "nll": value,
                         "tokens_scored": int(tokens.shape[1] - 1),
                         "effective_bits": row.bits, "slots": row.slots(dimensions["down"])}
                        for index, value in enumerate(values))
            base.write_csv(path, rows)
            LOG.info("E20 %s seed %d ppl=%.6f", row.name, seed_index, ppl)

    for name, payload in (("e20_round_trip_audit.csv", round_trips),
                          ("e20_alignment_diagnostic.csv", align_rows),
                          ("e20_f_of_config.csv", theory_rows),
                          ("e20_c_precision.csv", precision_rows)):
        if payload:
            base.write_csv(result_dir / name, payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    summarize(workdir, model_key, dimensions)


def summarize(workdir: Path, model_key: str, dimensions: dict[str, int]) -> None:
    path = per_sequence_path(workdir, model_key)
    if not path.exists():
        return
    raw = base.read_csv(path)
    by: dict[tuple[str, int], list[float]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for r in raw:
        by.setdefault((str(r["row"]), int(r["seed"])), []).append(float(r["nll"]))
        meta[str(r["row"])] = {"method": r["method"], "group": int(r["group"]),
                               "m": int(r["m"]), "effective_bits": float(r["effective_bits"]),
                               "slots": int(r["slots"])}
    ppl = {key: math.exp(float(np.mean(values))) for key, values in by.items()}
    names = sorted({name for name, _ in ppl})
    bf16 = float(np.mean([v for (n, _), v in ppl.items() if n == "bf16"])) if "bf16" in names else math.nan
    summary: list[dict[str, Any]] = []
    for name in names:
        seeds = sorted(seed for n, seed in ppl if n == name)
        values = [ppl[(name, seed)] for seed in seeds]
        base_values = [ppl[(BASELINE, seed)] for seed in seeds if (BASELINE, seed) in ppl]
        mean, low, high = (math.nan, math.nan, math.nan)
        if name != BASELINE and len(base_values) == len(values) and len(values) > 1:
            mean, low, high = paired_ci([v - b for v, b in zip(values, base_values)])
        summary.append({
            "model": model_key, "row": name, **meta[name], "seeds": len(values),
            "mean_ppl": float(np.mean(values)),
            "seed_ppl_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ppl_delta_vs_bf16": float(np.mean(values)) - bf16,
            "paired_delta_vs_nar_g128_m1": mean,
            "paired_90ci_low": low, "paired_90ci_high": high,
            "seed_ppls": " ".join(f"{v:.6f}" for v in values),
        })
    base.write_csv(workdir / "results" / model_key / "e20_summary.csv", summary)
    print(json.dumps(summary, indent=2), flush=True)


def finalize(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    payload: dict[str, Any] = {"models": {}, "rows": [r.name for r in ROWS],
                               "baseline": BASELINE, "eval_chunks": args.chunks,
                               "seeds": list(args.seeds), "fold": "exact_transpose",
                               "nll_dtype": "float32",
                               "hypotheses": {
                                   "H1": "NAR g256 m=2 within CI of or better than NAR g128 m=1",
                                   "H2": "NAR g256 m=3 beats NAR g128 m=1 at equal bits",
                                   "H3": "Hadamard g256 m=2/3 does not improve on Hadamard g256 m=1"},
                               "hardware": base.hardware_info()}
    for model_key in args.models:
        path = workdir / "results" / model_key / "e20_summary.csv"
        payload["models"][model_key] = base.read_csv(path) if path.exists() else None
    base.atomic_json(workdir / "results" / "llama32_3b" / "E20_DONE.json", payload)
    print(json.dumps({m: bool(v) for m, v in payload["models"].items()}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", required=True)
    result.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    result.add_argument("--rows", nargs="+", default=["all"])
    result.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    result.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    result.add_argument("--chunks", type=int, default=EVAL_CHUNKS)
    result.add_argument("--seq-len", type=int, default=2048)
    result.add_argument("--round-trip-tolerance", type=float, default=1e-6)
    result.add_argument("--finalize-only", action="store_true")
    return result


def run(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e20")
    if not args.finalize_only:
        if not torch.cuda.is_available():
            raise RuntimeError("E20 requires CUDA")
        for model_key in args.models:
            run_model(args, model_key)
    finalize(args)


if __name__ == "__main__":
    run(parser().parse_args())
