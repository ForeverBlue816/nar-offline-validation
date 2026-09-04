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
import collections
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
    coeff: str = "fp16"  # "fp32" rows are diagnostics, not bit-accounted

    @property
    def name(self) -> str:
        if self.method == "bf16":
            return "bf16"
        suffix = "" if self.coeff == "fp16" else "_c32"
        return f"{self.method}_g{self.group}_m{self.m}{suffix}"

    @property
    def coefficient_dtype(self) -> torch.dtype:
        return torch.float16 if self.coeff == "fp16" else torch.float32

    @property
    def bit_accounted(self) -> bool:
        """fp32 coefficients cost 32 bits per group, which the 4 + 16(m+1)/g
        accounting does not charge; those rows are diagnostics only."""
        return self.coeff == "fp16"

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
    # Diagnostics: identical rotations and chunks, c_j stored fp32 instead of
    # fp16. Not bit-accounted; they isolate coefficient precision.
    Row("nar", 128, 2, "fp32"), Row("nar", 256, 3, "fp32"),
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
        result = aq.quantize_affine(rotated, self.row.group, self.row.m,
                                    self.row.coefficient_dtype).dequant
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
                         top: int = 256) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            "directions_measured": int(min(top, vectors.shape[1])),
            "window_saturated": bool(row.slots(n) >= min(top, vectors.shape[1])),
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
                         "group": row.group, "m": row.m, "coeff": row.coeff,
                         "bit_accounted": row.bit_accounted, "seed": seed_index,
                         "chunk": index, "nll": value,
                         "tokens_scored": int(tokens.shape[1] - 1),
                         "effective_bits": row.bits, "slots": row.slots(dimensions["down"])}
                        for index, value in enumerate(values))
            base.write_csv(path, rows)
            LOG.info("E20 %s seed %d ppl=%.6f", row.name, seed_index, ppl)

    # Merge rather than overwrite: a run restricted to --rows only computes a
    # subset, and writing it wholesale would clobber the rows another
    # invocation produced. Keyed on the identifying columns of each table.
    for name, payload, keys in (
            ("e20_round_trip_audit.csv", round_trips, ("row", "seed", "site")),
            ("e20_alignment_diagnostic.csv", align_rows, ("row", "site", "layer", "direction")),
            ("e20_f_of_config.csv", theory_rows, ("row", "site")),
            ("e20_c_precision.csv", precision_rows, ("row", "site"))):
        if not payload:
            continue
        path = result_dir / name
        merged: dict[tuple, dict[str, Any]] = {}
        if path.exists():
            for existing in base.read_csv(path):
                merged[tuple(str(existing.get(k, "")) for k in keys)] = existing
        for fresh in payload:
            merged[tuple(str(fresh.get(k, "")) for k in keys)] = fresh
        base.write_csv(path, list(merged.values()))
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
                               "m": int(r["m"]), "coeff": r.get("coeff", "fp16"),
                               "bit_accounted": r.get("bit_accounted", "True"),
                               "effective_bits": float(r["effective_bits"]),
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
    result.add_argument("--range-only", action="store_true",
                        help="measured range reduction on the E1c dumps, plus the plot")
    result.add_argument("--range-rows", type=int, default=4096)
    result.add_argument("--range-layer-stride", type=int, default=1)
    result.add_argument("--range-directions", type=int, default=256)
    result.add_argument("--range-capture-sequences", type=int, default=4)
    result.add_argument("--f-directions", type=int, default=256)
    result.add_argument("--theory-only", action="store_true")
    result.add_argument("--two-term-only", action="store_true")
    return result


def run(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e20")
    if args.two_term_only:
        two_term_command(args)
        return
    if args.theory_only:
        if not torch.cuda.is_available():
            raise RuntimeError("E20 theory recomputation requires CUDA")
        theory_command(args)
        return
    if args.range_only:
        if not torch.cuda.is_available():
            raise RuntimeError("E20 range measurement requires CUDA")
        range_command(args)
        return
    if not args.finalize_only:
        if not torch.cuda.is_available():
            raise RuntimeError("E20 requires CUDA")
        for model_key in args.models:
            run_model(args, model_key)
    finalize(args)



# ------------------------------------------- measured range reduction (E1c) ---

E1C_SITE = {"qkv": "q_input", "down": "down_input"}
RANGE_MODEL = "llama32_3b"          # the model with frozen E1c dumps


def expected_max_gaussian(g: int, lo: float = -12.0, hi: float = 14.0,
                          points: int = 400001) -> float:
    """E[max of g iid N(0,1)], by quadrature on 1 - Phi^g.

    The group-size factor in the two-term predictor is this, not a fitted
    constant: for a rotated group the coordinates are approximately iid
    Gaussian, so the expected range is 2 E[max_g] and the step ratio between
    two group sizes is E[max_g2] / E[max_g1] with no free parameter.
    """
    x = np.linspace(lo, hi, points)
    phi = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    cdf = phi ** g
    positive = x >= 0
    return float(np.trapezoid(1.0 - cdf[positive], x[positive])
                 - np.trapezoid(cdf[~positive], x[~positive]))


@torch.inference_mode()
def capture_range_rows(workdir: Path, model_key: str, layers: list[int],
                       rows: int, sequences: int, seq_len: int
                       ) -> dict[tuple[str, int], torch.Tensor]:
    """Activations at the two sites for models without frozen E1c dumps."""
    model_id, _ = act.model_id_and_key(model_key)
    tokens = base.prepare_token_chunks(model_id, "train", 0, sequences, seq_len, workdir)
    model = base.load_model(model_id, workdir)
    store: dict[tuple[str, int], torch.Tensor] = {}
    handles: list[Any] = []

    def keep(site: str, layer: int, value: torch.Tensor) -> None:
        key = (site, layer)
        flat = value.detach().reshape(-1, value.shape[-1]).to("cpu", torch.bfloat16)
        if key not in store:
            store[key] = flat[:rows].clone()
        elif store[key].shape[0] < rows:
            store[key] = torch.cat((store[key], flat))[:rows].clone()

    for layer in layers:
        block = model.model.layers[layer]
        handles.append(block.input_layernorm.register_forward_hook(
            lambda _m, _i, out, layer=layer: keep("qkv", layer, out)))
        handles.append(block.mlp.down_proj.register_forward_pre_hook(
            lambda _m, inputs, layer=layer: keep("down", layer, inputs[0])))
    try:
        for index in range(tokens.shape[0]):
            model(input_ids=tokens[index:index + 1].cuda(), use_cache=False)
            if all(v.shape[0] >= rows for v in store.values()) and len(store) == 2 * len(layers):
                break
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return store


@torch.inference_mode()
def range_command(args: argparse.Namespace) -> None:
    """Measured range reduction on the E1c dump rows, against 1 - sqrt(1-f).

    Follows the E7 convention exactly: the response is the mean group range
    (max - min), which is already invariant to the DC component the zero-point
    absorbs, and the reference is the Hadamard rotation at the same group size.
    For m > 1 the extra directions are projected out before the range is taken,
    because that is what the affine quantizer's scale actually sees.
    """
    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e20-range")
    device = torch.device("cuda")
    for model_key in args.models:
        range_model(args, workdir, model_key, device)


def range_model(args: argparse.Namespace, workdir: Path, model_key: str,
                device: torch.device) -> None:
    from transformers import AutoConfig

    wide = workdir / "activations" / model_key / "wide_cal_a"
    use_dumps = (wide / "DONE.json").exists()
    if use_dumps:
        meta = json.loads((wide / "DONE.json").read_text())
        dimensions = {"qkv": int(meta["hidden_size"]), "down": int(meta["intermediate_size"])}
        layer_total = int(meta["num_layers"])
    else:
        model_id, _ = act.model_id_and_key(model_key)
        config = AutoConfig.from_pretrained(
            model_id, cache_dir=str(workdir / "cache" / "huggingface"))
        dimensions = {"qkv": int(config.hidden_size), "down": int(config.intermediate_size)}
        layer_total = int(config.num_hidden_layers)
        meta = {}
    layers = list(range(0, layer_total, args.range_layer_stride))
    captured = ({} if use_dumps
                else capture_range_rows(workdir, model_key, layers, args.range_rows,
                                        args.range_capture_sequences, args.seq_len))
    fractions = eigen_fractions(workdir, model_key)
    configs = [row for row in ROWS if row.method != "bf16"]
    rows: list[dict[str, Any]] = []

    for site, n in dimensions.items():
        for layer in layers:
            if use_dumps:
                mapped = act.ext._open_site(wide, meta, E1C_SITE[site], layer)
                x = act.ext._bits_to_tensor(
                    mapped.reshape(-1, n)[: args.range_rows], device).float()
            else:
                x = captured[(site, layer)].to(device).float()
            vectors, energy = load_site_data(workdir, model_key, site, layer, device)
            weights = torch.tensor(fractions[(site, layer)], device=device, dtype=torch.float32)
            signs = {(site, layer): torch.randint(
                0, 2, (n,), generator=torch.Generator(device="cpu").manual_seed(
                    args.seed + 1000 * layer + (0 if site == "qkv" else 100_000)),
            ).float().mul_(2).sub_(1).to(device)}
            hadamard = Rotation(Row("hadamard", 128, 1), {}, signs)
            had_x = hadamard.apply(site, layer, x)
            had_range = {g: base.mean_group_range(had_x, g) for g in (64, 128, 256)}
            del had_x

            for row in configs:
                g, m = row.group, row.m
                if row.method == "nar":
                    factor = build_nar_factor(vectors, energy, g, m)
                    rotation = Rotation(row, {(site, layer): factor}, signs)
                else:
                    rotation = Rotation(row, {}, signs)
                rotated = rotation.apply(site, layer, x)
                grouped = base.group_view(rotated, g)
                if m > 1:
                    basis = aq.walsh_basis(g, m, device)[1:]
                    grouped = grouped - (grouped @ basis.T) @ basis
                measured = float((grouped.amax(-1) - grouped.amin(-1)).mean())
                # f over the same P_N used by the quantizer, energy-weighted.
                projector = null_space_basis(n, g, m, device)
                take = min(args.range_directions, vectors.shape[1], weights.numel())
                share = rotation.apply(site, layer, vectors[:, :take].T)
                share = (share @ projector.T).square().sum(-1) / share.square().sum(-1).clamp_min(1e-30)
                f = float((share * weights[:take]).sum())
                rows.append({
                    "model": model_key, "row": row.name, "site": site, "layer": layer,
                    "group": g, "m": m, "slots": row.slots(n),
                    "effective_bits": row.bits, "rows_used": int(x.shape[0]),
                    "mean_group_range": measured,
                    "hadamard_reference_range": had_range[g],
                    "range_ratio_vs_hadamard": measured / had_range[g],
                    "measured_range_reduction": (had_range[g] - measured) / had_range[g],
                    "absorbed_energy_fraction": f,
                    "sqrt_one_minus_f": math.sqrt(max(0.0, 1.0 - f)),
                    "predicted_range_reduction": 1.0 - math.sqrt(max(0.0, 1.0 - f)),
                })
                del rotated, grouped, projector, share
            del x, vectors
            gc.collect()
            torch.cuda.empty_cache()
            LOG.info("E20 range %s layer %d done", site, layer)

    result_dir = workdir / "results" / model_key
    base.write_csv(result_dir / "e20_range_vs_config.csv", rows)
    base.write_csv(result_dir / "e20_step_ratio.csv", step_ratio_rows(rows, model_key))

    # E7's fit, pooled over layers, separately per m so the m>1 points can be
    # read against the m=1 line rather than averaged into it.
    fit_rows: list[dict[str, Any]] = []
    for key, label in (((1,), "m=1"), ((2,), "m=2"), ((3,), "m=3"), ((1, 2, 3), "all m")):
        subset = [r for r in rows if r["m"] in key and r["row"].startswith("nar")
                  and not r["row"].endswith("_c32")]
        if len(subset) < 3:
            continue
        predictor = np.asarray([r["sqrt_one_minus_f"] for r in subset])
        response = np.asarray([r["range_ratio_vs_hadamard"] for r in subset])
        design = np.column_stack((np.ones_like(predictor), predictor))
        intercept, slope = np.linalg.lstsq(design, response, rcond=None)[0]
        predicted = design @ np.asarray([intercept, slope])
        residual = float(np.square(response - predicted).sum())
        total = float(np.square(response - response.mean()).sum())
        fit_rows.append({
            "model": model_key, "subset": label, "points": len(subset),
            "intercept": float(intercept), "slope": float(slope),
            "r_squared": 1 - residual / total if total > 0 else math.nan,
            "rmse": math.sqrt(residual / len(response)),
            "mean_measured_reduction": float(np.mean([r["measured_range_reduction"] for r in subset])),
            "mean_predicted_reduction": float(np.mean([r["predicted_range_reduction"] for r in subset])),
            "fit": "OLS range/range_hadamard = intercept + slope*sqrt(1-f), pooled over layers and sites",
        })
    base.write_csv(result_dir / "e20_range_fit.csv", fit_rows)
    plot_range(rows, fit_rows, result_dir / "e20_range_vs_sqrt_one_minus_f.png")
    print(json.dumps(fit_rows, indent=2), flush=True)


def step_ratio_rows(rows: list[dict[str, Any]], model_key: str) -> list[dict[str, Any]]:
    """Measured Hadamard step ratio between group sizes against the analytic one."""
    had: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
    for r in rows:
        had[(str(r["site"]), int(r["group"]))].append(float(r["hadamard_reference_range"]))
    analytic = {g: expected_max_gaussian(g) for g in (64, 128, 256)}
    out: list[dict[str, Any]] = []
    for site in sorted({site for site, _ in had}):
        for big, small in ((256, 128), (128, 64)):
            if (site, big) not in had or (site, small) not in had:
                continue
            measured = float(np.mean(had[(site, big)])) / float(np.mean(had[(site, small)]))
            predicted = analytic[big] / analytic[small]
            out.append({
                "model": model_key, "site": site, "pair": f"{big}/{small}",
                "measured_hadamard_step_ratio": measured,
                "analytic_gaussian_ratio": predicted,
                "expected_max_numerator": analytic[big],
                "expected_max_denominator": analytic[small],
                "relative_error": (measured - predicted) / predicted,
                "source": "E[max of g iid N(0,1)] by quadrature; no fitted parameter",
            })
    return out


def plot_range(rows: list[dict[str, Any]], fit_rows: list[dict[str, Any]], path: Path) -> None:
    """The sqrt(1-f) plot with the m = 2 and m = 3 points added."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    styles = {1: ("o", "#1b4965"), 2: ("s", "#c1666b"), 3: ("^", "#48a9a6")}
    nar = [r for r in rows if r["row"].startswith("nar")]
    for m, (marker, colour) in styles.items():
        subset = [r for r in nar if r["m"] == m]
        if not subset:
            continue
        axes[0].scatter([r["sqrt_one_minus_f"] for r in subset],
                        [r["range_ratio_vs_hadamard"] for r in subset],
                        s=26, marker=marker, alpha=0.65, color=colour,
                        edgecolors="none", label=f"NAR m={m}")
    grid = np.linspace(0.6, 1.0, 50)
    axes[0].plot(grid, grid, "k--", linewidth=1.2, label="range ratio = sqrt(1-f)")
    fit = next((f for f in fit_rows if f["subset"] == "all m"), None)
    if fit:
        axes[0].plot(grid, fit["intercept"] + fit["slope"] * grid, color="#8d6a9f",
                     linewidth=1.6,
                     label=f"OLS all m (slope {fit['slope']:.2f}, R^2 {fit['r_squared']:.2f})")
    axes[0].set_xlabel(r"$\sqrt{1-f}$")
    axes[0].set_ylabel("mean group range / Hadamard reference")
    axes[0].set_title("E20: the law holds for m > 1")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.25, linewidth=0.5)

    for m, (marker, colour) in styles.items():
        subset = [r for r in nar if r["m"] == m]
        if not subset:
            continue
        axes[1].scatter([r["predicted_range_reduction"] for r in subset],
                        [r["measured_range_reduction"] for r in subset],
                        s=26, marker=marker, alpha=0.65, color=colour,
                        edgecolors="none", label=f"NAR m={m}")
    limit = max(0.05, max(r["measured_range_reduction"] for r in nar),
                max(r["predicted_range_reduction"] for r in nar))
    axes[1].plot([0, limit], [0, limit], "k--", linewidth=1.2, label="measured = predicted")
    axes[1].set_xlabel(r"predicted reduction $1-\sqrt{1-f}$")
    axes[1].set_ylabel("measured range reduction")
    axes[1].set_title("E20: predicted versus measured")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)

# --------------------------------------------- close-out diagnostics (E20) ---

def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without scipy; ties averaged."""
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out
    ra, rb = ranks(a), ranks(b)
    ma, mb = float(np.mean(ra)), float(np.mean(rb))
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else math.nan


@torch.inference_mode()
def theory_command(args: argparse.Namespace) -> None:
    """Recompute f with a wider measurement window and re-emit the theory CSV."""
    from transformers import AutoConfig

    workdir = Path(args.workdir).resolve()
    base.setup_logging(workdir, "e20-theory")
    device = torch.device("cuda")
    for model_key in args.models:
        model_id, _ = act.model_id_and_key(model_key)
        config = AutoConfig.from_pretrained(
            model_id, cache_dir=str(workdir / "cache" / "huggingface"))
        dimensions = {"qkv": int(config.hidden_size), "down": int(config.intermediate_size)}
        layers = int(config.num_hidden_layers)
        theory_rows: list[dict[str, Any]] = []
        align_rows: list[dict[str, Any]] = []
        for row in ROWS:
            if row.method == "bf16" or (row.method == "hadamard" and row.m == 1):
                continue
            if not row.bit_accounted:
                continue          # identical rotation to its fp16 twin
            rotation = build_rotation(workdir, model_key, row, dimensions, layers,
                                      args.seeds[0], args.seed, device)
            a, t = alignment_and_theory(workdir, model_key, row, rotation, dimensions,
                                        layers, device, top=args.f_directions)
            align_rows.extend(a)
            theory_rows.extend(t)
            LOG.info("E20 theory %s %s done", model_key, row.name)
            del rotation
            gc.collect()
            torch.cuda.empty_cache()
        result_dir = workdir / "results" / model_key
        base.write_csv(result_dir / "e20_f_of_config.csv", theory_rows)
        base.write_csv(result_dir / "e20_alignment_diagnostic.csv", align_rows)
        print(json.dumps([{k: r[k] for k in ("row", "site", "slots", "mean_captured_fraction",
                                             "directions_measured", "window_saturated")}
                          for r in theory_rows], indent=2), flush=True)


def two_term_command(args: argparse.Namespace) -> None:
    """Rank the NAR rows by f alone and by step_Hadamard(g) * sqrt(1-f).

    f is scale-free, so a ranking built on it cannot see that a 256-wide group
    starts from a larger step than a 128-wide one. The two-term predictor puts
    that back using the measured Hadamard step at the same group size.
    """
    workdir = Path(args.workdir).resolve()
    for model_key in args.models:
        if (workdir / "results" / model_key / "e20_range_vs_config.csv").exists():
            two_term_model(workdir, model_key)


def two_term_model(workdir: Path, model_key: str) -> None:
    result_dir = workdir / "results" / model_key
    range_rows = base.read_csv(result_dir / "e20_range_vs_config.csv")
    summary = {r["row"]: r for r in base.read_csv(result_dir / "e20_summary.csv")}

    per_site: dict[tuple[str, str], dict[str, float]] = {}
    for r in range_rows:
        # The six bit-accounted NAR rows only; the fp32-coefficient rows are
        # diagnostics on the same rotations and would double-count them.
        if not str(r["row"]).startswith("nar") or str(r["row"]).endswith("_c32"):
            continue
        key = (str(r["row"]), str(r["site"]))
        entry = per_site.setdefault(key, {"f": [], "had": [], "measured": []})
        entry["f"].append(float(r["absorbed_energy_fraction"]))
        entry["had"].append(float(r["hadamard_reference_range"]))
        entry["measured"].append(float(r["mean_group_range"]))

    rows: list[dict[str, Any]] = []
    names = sorted({name for name, _ in per_site})
    for name in names:
        combined_one, combined_two, combined_meas = [], [], []
        for site in ("qkv", "down"):
            entry = per_site[(name, site)]
            f = float(np.mean(entry["f"]))
            had_step = float(np.mean(entry["had"])) / 15.0
            step_pred = had_step * math.sqrt(max(0.0, 1.0 - f))
            step_meas = float(np.mean(entry["measured"])) / 15.0
            rows.append({
                "model": model_key, "row": name, "site": site,
                "group": int(summary[name]["group"]), "m": int(summary[name]["m"]),
                "slots": int(summary[name]["slots"]), "f": f,
                "hadamard_step": had_step, "one_term_score_f": f,
                "two_term_step_pred": step_pred, "measured_step": step_meas,
                "mean_ppl": float(summary[name]["mean_ppl"]),
            })
            combined_one.append(f)
            combined_two.append(step_pred)
            combined_meas.append(step_meas)
        rows.append({
            "model": model_key, "row": name, "site": "combined",
            "group": int(summary[name]["group"]), "m": int(summary[name]["m"]),
            "slots": int(summary[name]["slots"]), "f": float(np.mean(combined_one)),
            "hadamard_step": math.nan, "one_term_score_f": float(np.mean(combined_one)),
            "two_term_step_pred": float(np.mean(combined_two)),
            "measured_step": float(np.mean(combined_meas)),
            "mean_ppl": float(summary[name]["mean_ppl"]),
        })

    fits: list[dict[str, Any]] = []
    for site in ("qkv", "down", "combined"):
        subset = [r for r in rows if r["site"] == site]
        ppl = [r["mean_ppl"] for r in subset]
        # Lower PPL is better; higher f is better, lower step_pred is better.
        one = spearman([-r["one_term_score_f"] for r in subset], ppl)
        two = spearman([r["two_term_step_pred"] for r in subset], ppl)
        by_one = sorted(subset, key=lambda r: -r["one_term_score_f"])
        by_two = sorted(subset, key=lambda r: r["two_term_step_pred"])
        by_ppl = sorted(subset, key=lambda r: r["mean_ppl"])
        def position(ordering: list[dict[str, Any]], name: str) -> int:
            return [r["row"] for r in ordering].index(name) + 1
        fits.append({
            "model": model_key, "site": site, "rows": len(subset),
            "spearman_one_term_f": one, "spearman_two_term_step": two,
            "one_term_order": " > ".join(r["row"] for r in by_one),
            "two_term_order": " > ".join(r["row"] for r in by_two),
            "measured_ppl_order": " > ".join(r["row"] for r in by_ppl),
            "one_term_rank_g256_m3": position(by_one, "nar_g256_m3"),
            "one_term_rank_g128_m1": position(by_one, "nar_g128_m1"),
            "two_term_rank_g256_m3": position(by_two, "nar_g256_m3"),
            "two_term_rank_g128_m1": position(by_two, "nar_g128_m1"),
            "measured_rank_g256_m3": position(by_ppl, "nar_g256_m3"),
            "measured_rank_g128_m1": position(by_ppl, "nar_g128_m1"),
            "one_term_places_g256_m3_above_g128_m1":
                position(by_one, "nar_g256_m3") < position(by_one, "nar_g128_m1"),
            "two_term_places_g256_m3_above_g128_m1":
                position(by_two, "nar_g256_m3") < position(by_two, "nar_g128_m1"),
            "measured_places_g256_m3_above_g128_m1":
                position(by_ppl, "nar_g256_m3") < position(by_ppl, "nar_g128_m1"),
        })
    base.write_csv(result_dir / "e20_two_term_theory.csv", rows)
    base.write_csv(result_dir / "e20_two_term_ranking.csv", fits)
    print(json.dumps([{k: f[k] for k in ("model", "site", "spearman_one_term_f",
                                         "spearman_two_term_step",
                                         "one_term_places_g256_m3_above_g128_m1",
                                         "two_term_places_g256_m3_above_g128_m1",
                                         "two_term_order", "measured_ppl_order")}
                      for f in fits], indent=2), flush=True)


if __name__ == "__main__":
    run(parser().parse_args())
