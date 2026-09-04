#!/usr/bin/env python3
"""Prepare the BOS-excluded eigenspace and geometry inputs for Figure 3."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


SEED = 20260902
OVERSAMPLE = 16
RANK = 256
GROUP = 128
LAYERS = (1, 13, 27)
PROJECTION_STRIDE = 32


def import_project(repo: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(repo))
    from nar import activation_experiments as act
    from nar import extended_experiment as ext

    return act, ext


def iter_non_bos(
    mmap: np.memmap,
    ext: Any,
    device: torch.device,
    sequence_batch: int,
) -> Iterable[torch.Tensor]:
    for start in range(0, mmap.shape[0], sequence_batch):
        stop = min(start + sequence_batch, mmap.shape[0])
        bits = mmap[start:stop, 1:, :]
        yield ext._bits_to_tensor(bits, device).reshape(-1, mmap.shape[-1])


def covariance_apply(
    mmap: np.memmap,
    q: torch.Tensor,
    ext: Any,
    device: torch.device,
    sequence_batch: int,
    compute_trace: bool = False,
) -> tuple[torch.Tensor, float | None, int]:
    q_device = q.to(device=device, dtype=torch.float32)
    result = torch.zeros_like(q, dtype=torch.float64, device="cpu")
    trace_sum = 0.0
    count = 0
    for x in iter_non_bos(mmap, ext, device, sequence_batch):
        projected = x @ q_device
        result += (x.T @ projected).double().cpu()
        if compute_trace:
            trace_sum += float(x.square().sum(dtype=torch.float64).item())
        count += x.shape[0]
        del x, projected
    result /= count
    return result, (trace_sum / count if compute_trace else None), count


def solve(
    mmap: np.memmap,
    layer: int,
    ext: Any,
    device: torch.device,
    sequence_batch: int,
) -> dict[str, Any]:
    n = int(mmap.shape[-1])
    width = RANK + OVERSAMPLE
    generator = torch.Generator(device="cpu").manual_seed(SEED + 100_000 + layer)
    omega = torch.randn((n, width), generator=generator, dtype=torch.float64)
    q = torch.linalg.qr(omega, mode="reduced").Q.float()
    for _ in range(2):
        cq, _trace, _rows = covariance_apply(mmap, q, ext, device, sequence_batch)
        q = torch.linalg.qr(cq, mode="reduced").Q.float()
    cq, trace, rows = covariance_apply(
        mmap, q, ext, device, sequence_batch, compute_trace=True
    )
    small = q.double().T @ cq.double()
    small = (small + small.T) / 2
    evals, u = torch.linalg.eigh(small)
    order = torch.argsort(evals, descending=True)[:RANK]
    evals = evals[order].clamp_min(0)
    u = u[:, order]
    vectors = q.double() @ u
    residual = cq.double() @ u - vectors * evals.unsqueeze(0)
    relative = residual.norm(dim=0) / evals.clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "vectors": vectors.float(),
        "eigenvalues": evals,
        "trace": float(trace),
        "relative_residuals": relative,
        "rows": rows,
    }


def sampled_rows(
    mmap: np.memmap,
    ext: Any,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    positions = np.arange(PROJECTION_STRIDE, mmap.shape[1], PROJECTION_STRIDE, dtype=np.int64)
    sequences = np.repeat(np.arange(mmap.shape[0], dtype=np.int64), len(positions))
    token_positions = np.tile(positions, mmap.shape[0])
    x = ext._bits_to_tensor(mmap[:, positions, :], device).reshape(-1, mmap.shape[-1])
    if torch.any(torch.as_tensor(token_positions, device=x.device) == 0):
        raise AssertionError("BOS leaked into Figure 3 projection rows")
    return x.float(), sequences, token_positions


def inverse_factor(y: torch.Tensor, factor: Any, ext: Any, signs: torch.Tensor) -> torch.Tensor:
    rows = y.reshape(-1, factor.n // factor.b, factor.b)
    permuted = ext._fast_walsh_hadamard(rows).reshape(-1, factor.n) * signs
    before_g = torch.empty_like(permuted)
    before_g[:, factor.source_order] = permuted[:, factor.target_order]
    for index in range(factor.reflectors.shape[0] - 1, -1, -1):
        if bool(factor.active[index]):
            vector = factor.reflectors[index]
            before_g -= 2 * (before_g @ vector).unsqueeze(1) * vector.unsqueeze(0)
    return before_g.reshape_as(y)


def geometry(
    vectors: torch.Tensor,
    x: torch.Tensor,
    sequences: np.ndarray,
    positions: np.ndarray,
    act: Any,
    ext: Any,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rank = 8192 // GROUP
    factor = act.factor_from_vectors(vectors[:, :rank].to(device), x, GROUP)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1000 * 1 + 10 + GROUP)
    signs = torch.randint(0, 2, (8192,), generator=generator, dtype=torch.int64)
    signs = signs.float().mul_(2).sub_(1).to(device)

    source_index = int((factor.source_order == 0).nonzero(as_tuple=False).item())
    target_coordinate = int(factor.target_order[source_index])
    target_group = target_coordinate // GROUP
    null_vector = torch.zeros((1, 8192), device=device)
    null_vector[:, target_group * GROUP : (target_group + 1) * GROUP] = 1 / math.sqrt(GROUP)

    prism_preimage = inverse_factor(null_vector, factor, ext, signs).squeeze(0)
    had_preimage = (
        ext._fast_walsh_hadamard(null_vector.reshape(-1, GROUP)).reshape_as(null_vector)
        * signs
    ).squeeze(0)
    v = vectors[:, :2].to(device)
    projection = x @ v
    frame = pd.DataFrame(
        {
            "model": "llama32_3b",
            "site": "down_input",
            "layer": 1,
            "sequence_index": sequences,
            "token_position": positions,
            "projection_v1": projection[:, 0].float().cpu().numpy(),
            "projection_v2": projection[:, 1].float().cpu().numpy(),
            "bos_excluded": True,
        }
    )
    arrows: dict[str, Any] = {}
    for method, direction in (("hadamard", had_preimage), ("nar", prism_preimage)):
        coordinates = direction @ v
        arrows[method] = {
            "projection_v1": float(coordinates[0]),
            "projection_v2": float(coordinates[1]),
            "cosine_with_v1": float(torch.dot(direction, v[:, 0])),
            "in_plane_length": float(torch.linalg.vector_norm(coordinates)),
            "norm": float(torch.linalg.vector_norm(direction)),
        }
    metadata = {
        "target_group": target_group,
        "target_coordinate": target_coordinate,
        "anchor_error": float(factor.anchor_error),
        "arrows": arrows,
    }
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sequence-batch", type=int, default=4)
    args = parser.parse_args()

    repo = args.repo.resolve()
    workdir = args.workdir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    act, ext = import_project(repo)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    wide = workdir / "activations" / "llama32_3b" / "wide_cal_a"
    meta = json.loads((wide / "DONE.json").read_text())

    eigenspace_rows: list[dict[str, Any]] = []
    geometry_meta: dict[str, Any] | None = None
    for layer in LAYERS:
        mmap = ext._open_site(wide, meta, "down_input", layer)
        solution = solve(mmap, layer, ext, device, args.sequence_batch)
        values = solution["eigenvalues"]
        cumulative = torch.cumsum(values, dim=0) / float(solution["trace"])
        for index in range(RANK):
            eigenspace_rows.append(
                {
                    "model": "llama32_3b",
                    "site": "down_input",
                    "layer": layer,
                    "rank": index + 1,
                    "eigenvalue": float(values[index]),
                    "fraction_total_energy": float(values[index]) / float(solution["trace"]),
                    "cumulative_fraction_total_energy": float(cumulative[index]),
                    "relative_ritz_residual": float(solution["relative_residuals"][index]),
                    "rows_used": int(solution["rows"]),
                    "bos_excluded": True,
                    "seed": SEED + 100_000 + layer,
                    "oversample": OVERSAMPLE,
                    "covariance_passes": 3,
                }
            )
        if layer == 1:
            x, sequences, positions = sampled_rows(mmap, ext, device)
            projections, geometry_meta = geometry(
                solution["vectors"], x, sequences, positions, act, ext, device
            )
            projections.to_csv(output / "fig3_token_projections.csv", index=False)
            np.savez_compressed(
                output / "fig3_eigvecs_layer1.npz",
                vectors=solution["vectors"].numpy(),
                eigenvalues=values.numpy(),
                relative_ritz_residuals=solution["relative_residuals"].numpy(),
            )
        del mmap, solution
        torch.cuda.empty_cache()

    pd.DataFrame(eigenspace_rows).to_csv(output / "fig3_eigenspace_r256.csv", index=False)
    if geometry_meta is None:
        raise AssertionError("layer 1 geometry was not prepared")
    geometry_meta.update(
        {
            "model": "llama32_3b",
            "site": "down_input",
            "layer": 1,
            "dimensions": 8192,
            "group_size": GROUP,
            "null_space_slots": 64,
            "rank": RANK,
            "layers": list(LAYERS),
            "projection_rows": int(128 * len(range(PROJECTION_STRIDE, 2048, PROJECTION_STRIDE))),
            "projection_stride": PROJECTION_STRIDE,
            "eigensolver_rows": int(128 * 2047),
            "bos_exclusion_rule": "exclude sequence position 0 before all covariance and projection operations",
            "base_seed": SEED,
            "oversample": OVERSAMPLE,
            "power_iterations": 1,
            "covariance_passes": 3,
            "source": "frozen E1c wide_cal_a down_input dump",
        }
    )
    (output / "fig3_geometry_metadata.json").write_text(json.dumps(geometry_meta, indent=2) + "\n")
    print(json.dumps(geometry_meta, indent=2))


if __name__ == "__main__":
    main()
