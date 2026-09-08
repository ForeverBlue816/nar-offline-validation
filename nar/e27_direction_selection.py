#!/usr/bin/env python3
"""E27: preregistered full-token versus extreme-token subspace selection."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import activation_experiments as act, activation_diagnostics as diag
    from . import experiment as base, extended_experiment as ext
except ImportError:
    import activation_experiments as act, activation_diagnostics as diag
    import experiment as base, extended_experiment as ext

LOG = logging.getLogger('nar')
SEED, GROUP, LENGTH, NCAL, NEVAL, STRIDE = 20260902, 128, 2048, 128, 64, 32
VARIANTS = ('A_full', 'B_top1', 'B_top1_no_bos', 'C_top1pct', 'C_top1pct_no_bos')
CONDITIONS = ('qkv_only', 'both', 'down_only')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    return base.read_csv(path) if Path(path).exists() else []


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.csv.tmp')
    base.write_csv(temp, rows)
    os.replace(temp, path)


def roots(args, model):
    return args.workdir / 'activations' / model / 'e27', args.workdir / 'results' / model


def protocol(args):
    path = args.repo / 'experiments/e27_preregistration.json'
    p = read_json(path)
    assert p['top_p_scope'] == 'global per layer/site'
    assert p['rotation_seeds'] == [SEED, SEED + 1, SEED + 2]
    return p, digest(path)


def audit_inputs(args, model):
    pre, signature = protocol(args)
    assets = pre['assets'][model]
    for spec in assets['tokens'].values():
        path = args.workdir / spec['file']
        if digest(path) != spec['sha256']:
            raise RuntimeError(f'Frozen token cache changed: {path}')
    for name, sha in assets['A_factors_sha256'].items():
        if digest(act.factor_dir(args.workdir, model) / name) != sha:
            raise RuntimeError(f'Frozen A factor changed: {model}/{name}')
    train = torch.load(args.workdir / assets['tokens']['train']['file'], weights_only=True)
    test = torch.load(args.workdir / assets['tokens']['test']['file'], weights_only=True)
    assert train.shape == (NCAL, LENGTH) and test.shape == (NEVAL, LENGTH)
    assert bool((train[:, 0] == train[0, 0]).all())
    assert not bool((train[:, 1:] == train[0, 0]).any()), 'BOS exists beyond position zero; revisit candidate mask before running'
    return train, test, signature


def rank_indices(scores, flat_indices, count):
    """Stable descending score; earliest sequence/token breaks every tie."""
    scores, flat_indices = np.asarray(scores), np.asarray(flat_indices)
    if not np.isfinite(scores).all():
        raise RuntimeError('Nonfinite token selection norm')
    return np.lexsort((flat_indices, -scores))[:count]


def selection_indices(scores):
    nseq, length = scores.shape
    flat = scores.reshape(-1)
    ids = np.arange(flat.size)
    nonbos = ids[ids % length != 0]
    return {
        'B_top1': np.arange(nseq) * length + scores.argmax(1),
        'B_top1_no_bos': np.arange(nseq) * length + 1 + scores[:, 1:].argmax(1),
        'C_top1pct': ids[rank_indices(flat, ids, math.ceil(.01 * ids.size))],
        'C_top1pct_no_bos': nonbos[rank_indices(flat[nonbos], nonbos, math.ceil(.01 * nonbos.size))],
    }


class Capture:
    """One model pass; retain only global candidates, per-sequence maxima, and E1c rows."""
    def __init__(self, model, folder):
        self.model, self.folder, self.start = model, folder, 0
        self.handles, self.scores, self.pool, self.best, self.maps = [], {}, {}, {}, {}
        self.layers = len(model.model.layers)
        self.dimensions = {'qkv': model.config.hidden_size, 'down': model.config.intermediate_size}
        self.keep = math.ceil(.01 * NCAL * LENGTH) + NCAL
        for site, n in self.dimensions.items():
            for layer in range(self.layers):
                key = (site, layer)
                self.scores[key] = np.empty((NCAL, LENGTH), np.float32)
                self.pool[key] = (np.empty(0, np.float32), np.empty(0, np.int64), torch.empty((0, n), dtype=torch.bfloat16))
                self.best[key] = {}
                path = folder / f'{site}_layer_{layer:02d}.bf16'
                self.maps[key] = np.memmap(path, mode='w+', dtype=np.uint16, shape=(NCAL, LENGTH // STRIDE, n))

    def consume(self, site, layer, value):
        key = (site, layer)
        value = value.detach()
        assert value.shape[1] == LENGTH and value.dtype == torch.bfloat16
        batch, _, n = value.shape
        norms = value.abs().amax(-1).float().cpu().numpy()
        if not np.isfinite(norms).all():
            raise RuntimeError('Nonfinite calibration activations')
        self.scores[key][self.start:self.start + batch] = norms
        sampled = value[:, ::STRIDE].contiguous().cpu().view(torch.uint16).numpy()
        self.maps[key][self.start:self.start + batch] = sampled
        for offset in range(batch):
            for pos in (int(norms[offset].argmax()), int(norms[offset, 1:].argmax()) + 1):
                self.best[key][(self.start + offset) * LENGTH + pos] = value[offset, pos].cpu().clone()
        old_scores, old_ids, old_values = self.pool[key]
        new_ids = np.arange(self.start * LENGTH, (self.start + batch) * LENGTH)
        scores = np.concatenate([old_scores, norms.reshape(-1)])
        ids = np.concatenate([old_ids, new_ids])
        take = rank_indices(scores, ids, min(self.keep, len(ids)))
        rows = torch.empty((len(take), n), dtype=torch.bfloat16)
        old_mask = take < len(old_ids)
        rows[torch.from_numpy(np.flatnonzero(old_mask))] = old_values[torch.from_numpy(take[old_mask])]
        fresh = take[~old_mask] - len(old_ids)
        rows[torch.from_numpy(np.flatnonzero(~old_mask))] = value.reshape(-1, n)[torch.as_tensor(fresh, device=value.device)].cpu()
        self.pool[key] = (scores[take], ids[take], rows)

    def install(self):
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.input_layernorm.register_forward_hook(
                lambda module, inputs, output, layer=layer: self.consume('qkv', layer, output)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(
                lambda module, inputs, layer=layer: self.consume('down', layer, inputs[0])))

    def close(self):
        for handle in self.handles:
            handle.remove()
        for mmap in self.maps.values():
            mmap.flush()

    def finish(self, model_key):
        audit = []
        for key, scores in self.scores.items():
            site, layer = key
            _, pool_ids, pool_values = self.pool[key]
            lookup = {int(i): j for j, i in enumerate(pool_ids)}
            payload = {'norms': torch.from_numpy(scores), 'variants': {}}
            for method, ids in selection_indices(scores).items():
                values = torch.stack([pool_values[lookup[int(i)]] if int(i) in lookup else self.best[key][int(i)] for i in ids])
                payload['variants'][method] = {'indices': torch.from_numpy(ids), 'values': values}
                audit.append({'model': model_key, 'site': site, 'layer': layer, 'method': method,
                              'selected_tokens': len(ids), 'selected_bos': int((ids % LENGTH == 0).sum()),
                              'selected_sequences': len(np.unique(ids // LENGTH)),
                              'bos_fraction': float((ids % LENGTH == 0).mean()),
                              'indices_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                              'values_sha256': hashlib.sha256(values.view(torch.uint16).numpy().tobytes()).hexdigest(),
                              'selection_artifact': f'{site}_layer_{layer:02d}.selection.pt'})
            base.atomic_torch_save(self.folder / f'{site}_layer_{layer:02d}.selection.pt', payload)
            del payload
        return audit


def capture(args, model_key):
    train, _, signature = audit_inputs(args, model_key)
    folder, results = roots(args, model_key)
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'CAPTURE_DONE.json'
    if done.exists():
        assert read_json(done)['protocol_sha256'] == signature
        return
    model = base.load_model(act.MODEL_IDS[model_key], args.workdir)
    collector = Capture(model, folder)
    collector.install()
    try:
        with torch.inference_mode():
            for start in range(0, NCAL, 2):
                collector.start = start
                model.model(input_ids=train[start:start + 2].cuda(), use_cache=False)
                if (start + 2) % 8 == 0:
                    LOG.info('E27 capture %s %d/%d', model_key, start + 2, NCAL)
    finally:
        collector.close()
    rows = collector.finish(model_key)
    write_csv(results / 'e27_selection_audit.csv', rows)
    base.atomic_json(done, {'protocol_sha256': signature, 'completed_at_utc': utc(),
                           'layers': collector.layers, 'dimensions': collector.dimensions,
                           'calibration_sequences': NCAL, 'sequence_length': LENGTH,
                           'diagnostic_positions': list(range(0, LENGTH, STRIDE)),
                           'diagnostic_rows_per_layer_site': NCAL * (LENGTH // STRIDE),
                           'capture': 'bf16 replay of hashed frozen token IDs; raw E1c dump files unavailable',
                           'hardware': base.hardware_info()})
    del collector, model
    gc.collect()
    torch.cuda.empty_cache()


def recover_basis(factor):
    k = factor.n // factor.b
    anchors = torch.zeros((k, factor.n), device=factor.reflectors.device)
    anchors[torch.arange(k), torch.arange(k) * factor.b] = 1
    basis = act.apply_reflectors(anchors, factor.reflectors.flip(0), factor.active.flip(0)).T
    error = float((basis.T @ basis - torch.eye(k, device=basis.device)).abs().max())
    if error > 5e-5:
        raise RuntimeError(f'A recovered basis not orthonormal: {error}')
    return basis


def complete_basis(vectors, rank, seed):
    if vectors.shape[1] >= rank:
        return vectors[:, :rank]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    random = torch.randn((vectors.shape[0], rank - vectors.shape[1]), generator=generator, dtype=torch.float64).to(vectors.device)
    v = vectors.double()
    for _ in range(2):
        random -= v @ (v.T @ random)
    extra = torch.linalg.qr(random, mode='reduced').Q
    return torch.cat([v, extra], dim=1).float()


def estimate_basis(x, rank, seed, exact):
    x = x.float()
    if exact:
        _, singular, vh = torch.linalg.svd(x.double(), full_matrices=False)
        valid = singular > singular[0] * 1e-7
        effective = int(valid.sum())
        vectors = vh[valid].T[:, :rank]
        values = singular[valid][:rank].square() / len(x)
        vectors = complete_basis(vectors, rank, seed)
        info = {'estimator': 'thin float64 SVD of uncentered selected rows', 'selected_numerical_rank': effective,
                'null_completion_directions': max(0, rank - effective)}
    else:
        n = x.shape[1]
        generator = torch.Generator(device='cpu').manual_seed(seed)
        q = torch.linalg.qr(torch.randn((n, min(n, rank + 16)), generator=generator, dtype=torch.float64), mode='reduced').Q.float().to(x.device)
        for _ in range(2):
            cq = (x.T @ (x @ q)).double() / len(x)
            q = torch.linalg.qr(cq, mode='reduced').Q.float()
        cq = (x.T @ (x @ q)).double() / len(x)
        small = q.double().T @ cq
        values, u = torch.linalg.eigh((small + small.T) / 2)
        order = torch.argsort(values, descending=True)[:rank]
        values = values[order].clamp_min(0)
        vectors = (q.double() @ u[:, order]).float()
        info = {'estimator': 'fixed randomized second moment, oversample16, power1, Ritz',
                'selected_numerical_rank': -1, 'null_completion_directions': 0}
    orthogonal_error = float((vectors.T @ vectors - torch.eye(rank, device=x.device)).abs().max())
    if orthogonal_error > 5e-5:
        raise RuntimeError(f'Estimated basis error: {orthogonal_error}')
    return vectors.float(), {**info, 'orthogonality_max_abs_error': orthogonal_error,
                              'retained_eigenvalues': values.cpu(), 'rank': rank}


def layer_keys(args, model):
    folder, _ = roots(args, model)
    meta = read_json(folder / 'CAPTURE_DONE.json')
    return [(site, layer, n) for site, n in meta['dimensions'].items() for layer in range(meta['layers'])]


def evaluation_rows(args, model, site, layer, n):
    folder, _ = roots(args, model)
    path = folder / f'{site}_layer_{layer:02d}.bf16'
    shape = (NCAL, LENGTH // STRIDE, n)
    ext._validate_dump_file(path, shape)
    mmap = np.memmap(path, mode='r', dtype=np.uint16, shape=shape)
    return ext._bits_to_tensor(np.asarray(mmap), torch.device('cuda')).reshape(-1, n)


def factor_path(args, model, method, site, layer):
    if method == 'A_full':
        return act.factor_dir(args.workdir, model) / f'{site}_layer_{layer:02d}.pt'
    return roots(args, model)[0] / 'factors' / method / f'{site}_layer_{layer:02d}.pt'


def build(args, model, methods=VARIANTS[1:]):
    folder, results = roots(args, model)
    audit = csv_rows(results / 'e27_factor_audit.csv')
    for site, layer, n in layer_keys(args, model):
        original = act.RotationFactor.load(factor_path(args, model, 'A_full', site, layer), torch.device('cuda'))
        basis_a = recover_basis(original)
        selection = None
        for method in methods:
            path = factor_path(args, model, method, site, layer)
            if path.exists():
                continue
            seed = SEED + 100000 * (site == 'down') + layer
            if method == 'D_range_direct':
                vectors, extra = refine(args, model, site, layer, n, original, basis_a)
            else:
                if selection is None:
                    selection = torch.load(folder / f'{site}_layer_{layer:02d}.selection.pt', weights_only=True)
                x = selection['variants'][method]['values'].cuda().float()
                vectors, extra = estimate_basis(x, n // GROUP, seed, exact=method.startswith('B_'))
                extra['selected_tokens'] = len(x)
                del x
            refs, active, error = act.reflectors_from_vectors(vectors, GROUP)
            if error > 5e-5:
                raise RuntimeError(f'Householder anchor error {error}')
            factor = act.RotationFactor(n, GROUP, refs, active, original.source_order, original.target_order, error)
            probe = vectors.T
            mapped = factor.apply(probe, torch.ones(n, device='cuda')).reshape(n // GROUP, n // GROUP, GROUP)
            if float((mapped - mapped.mean(-1, keepdim=True)).abs().max()) > 5e-5:
                raise RuntimeError('Subspace does not map to group-constant directions')
            extra.update({'vectors': vectors.cpu(), 'method': method, 'site': site, 'layer': layer,
                          'model': model, 'source_A_sha256': digest(factor_path(args, model, 'A_full', site, layer)),
                          'subspace_seed': seed, 'completed_at_utc': utc()})
            path.parent.mkdir(parents=True, exist_ok=True)
            factor.save(path, extra)
            audit.append({'model': model, 'method': method, 'site': site, 'layer': layer,
                          'k': n // GROUP, 'anchor_error': error,
                          'selected_numerical_rank': extra.get('selected_numerical_rank', -1),
                          'null_completion_directions': extra.get('null_completion_directions', 0),
                          'permutation_identical_to_A': True, 'factor_sha256': digest(path)})
            write_csv(results / 'e27_factor_audit.csv', audit)
            LOG.info('E27 factor %s %s %s layer=%d rank=%d', model, method, site, layer, n // GROUP)
        del original, basis_a, selection
        gc.collect()
        torch.cuda.empty_cache()


def seed_signs(n, seed_index, site, layer, device='cuda'):
    gen = torch.Generator(device='cpu').manual_seed(act._seed(SEED, seed_index, layer, site))
    return torch.randint(0, 2, (n,), generator=gen).float().mul_(2).sub_(1).to(device)


def angles_degrees(a, b):
    qa = torch.linalg.qr(a[:, :8].double(), mode='reduced').Q
    qb = torch.linalg.qr(b[:, :8].double(), mode='reduced').Q
    return torch.rad2deg(torch.acos(torch.linalg.svdvals(qa.T @ qb).clamp(0, 1))).cpu().tolist()


def diagnostics(args, model, methods=(*VARIANTS, 'hadamard')):
    folder, results = roots(args, model)
    output = results / 'e27_f_and_angles.partial.csv'
    rows = csv_rows(output)
    completed = {(r['site'], int(r['layer']), r['method'], int(r['seed'])) for r in rows}
    for site, layer, n in layer_keys(args, model):
        needed = [(m, s) for m in methods for s in range(3) if (site, layer, m, SEED + s) not in completed]
        if not needed:
            continue
        x = evaluation_rows(args, model, site, layer, n)
        a_factor = act.RotationFactor.load(factor_path(args, model, 'A_full', site, layer), torch.device('cuda'))
        basis_a = recover_basis(a_factor)
        total_energy = float(x.float().square().sum(dtype=torch.float64))
        had_metrics = {}
        for si in range(3):
            signs = seed_signs(n, si, site, layer)
            sums = {}
            for begin in range(0, len(x), 512):
                rotated = act.full_hadamard_rows(x[begin:begin + 512].float(), signs)
                ext._merge_sums(sums, ext._quant_sums(rotated, GROUP))
                sums['capture_sum'] = sums.get('capture_sum', 0.) + float(rotated.reshape(-1, n // GROUP, GROUP).mean(-1).square().sum(dtype=torch.float64)) * GROUP
            had_metrics[si] = sums
        for method in methods:
            factor = None if method == 'hadamard' else act.RotationFactor.load(factor_path(args, model, method, site, layer), torch.device('cuda'))
            basis = None if factor is None else recover_basis(factor)
            angles = [math.nan] * 8 if basis is None else angles_degrees(basis_a, basis)
            identified_rank = -1
            if method == 'A_full':
                identified_rank = n // GROUP
            elif factor is not None:
                payload = torch.load(factor_path(args, model, method, site, layer), map_location='cpu', weights_only=True)
                identified_rank = int(payload.get('selected_numerical_rank', -1))
                del payload
            projected = math.nan if basis is None else float((x.float() @ basis).square().sum(dtype=torch.float64)) / total_energy
            for si in range(3):
                if (method, si) not in needed:
                    continue
                sums = had_metrics[si].copy() if factor is None else {}
                if factor is not None:
                    signs = seed_signs(n, si, site, layer)
                    for begin in range(0, len(x), 512):
                        rotated = factor.apply(x[begin:begin + 512], signs)
                        ext._merge_sums(sums, ext._quant_sums(rotated, GROUP))
                        sums['capture_sum'] = sums.get('capture_sum', 0.) + float(rotated.reshape(-1, n // GROUP, GROUP).mean(-1).square().sum(dtype=torch.float64)) * GROUP
                f = sums['capture_sum'] / sums['energy_sum']
                if not (0 <= f <= 1 + 1e-6):
                    raise RuntimeError(f'Invalid captured fraction {f}')
                if factor is not None and abs(projected - f) > 5e-5:
                    raise RuntimeError(f'Projection versus group capture disagrees: {projected}, {f}')
                rg = sums['range_sum'] / sums['group_count']
                nmse = sums['error_sum'] / sums['energy_sum']
                h = had_metrics[si]
                hr = h['range_sum'] / h['group_count']
                rows.append({'model': model, 'site': site, 'layer': layer, 'method': method, 'seed': SEED + si,
                             'n': n, 'k': n // GROUP, 'group_size': GROUP,
                             'evaluation_tokens': len(x), 'evaluation_rule': 'E1c positions 0:2048:32 in all 128 frozen sequences',
                             'f': f, 'subspace_projection_fraction': projected,
                             'selected_numerical_rank': identified_rank,
                             'top8_angles_include_null_completion': method.startswith('B_') and identified_rank < 8,
                             'sqrt_one_minus_f': math.sqrt(max(0, 1 - f)),
                             'mean_group_range': rg, 'relative_quantization_error_nmse': nmse,
                             'paired_hadamard_range': hr, 'range_over_paired_hadamard': rg / hr,
                             'range_reduction_vs_hadamard': 1 - rg / hr,
                             'nmse_delta_vs_hadamard': nmse - h['error_sum'] / h['energy_sum'],
                             **{f'angle_{j + 1}_degrees_vs_A_top8': v for j, v in enumerate(angles)},
                             'angles_mean_degrees_vs_A_top8': float(np.mean(angles)),
                             'factor_file': '' if factor is None else str(factor_path(args, model, method, site, layer).relative_to(args.workdir))})
            write_csv(output, rows)
            LOG.info('E27 diagnostics %s %s %s layer=%d', model, method, site, layer)
        del x, a_factor, basis_a, factor, basis
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(results / 'e27_f_and_angles.csv', rows)
    write_csv(results / 'e27_per_layer.csv', rows)


def refine(args, model, site, layer, n, original, basis_a):
    x = evaluation_rows(args, model, site, layer, n)
    gen = torch.Generator(device='cpu').manual_seed(SEED + 100000 * (site == 'down') + layer)
    batches = torch.randint(0, len(x), (200, 128), generator=gen)
    signs = seed_signs(n, 0, site, layer)
    current = basis_a.detach().clone()
    losses = []
    for step in range(200):
        current.requires_grad_(True)
        transformed = diag.differentiable_nar(x[batches[step].to(x.device)], current, original.source_order, original.target_order, signs)
        loss = diag.range_surrogate(transformed, 8)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite E8 counterpart objective')
        gradient = torch.autograd.grad(loss, current)[0]
        tangent = gradient - current @ (current.T @ gradient)
        tangent /= tangent.norm().clamp_min(1e-12)
        with torch.no_grad():
            updated = torch.linalg.qr(current - .05 * tangent, mode='reduced').Q
            alignment = torch.sign(torch.diag(updated.T @ current))
            alignment[alignment == 0] = 1
            current = updated * alignment
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0:
            LOG.info('E27 D %s %s layer=%d step=%d objective=%.6g', model, site, layer, step + 1, losses[-1])
    return current.detach(), {'step_losses': losses, 'steps': 200, 'p': 8, 'learning_rate': .05,
                              'batch_size': 128, 'selected_numerical_rank': -1, 'null_completion_directions': 0}


class Rotations:
    def __init__(self, args, model, method, seed_index):
        self.method, self.factors, self.signs = method, {}, {}
        for site, layer, n in layer_keys(args, model):
            self.signs[(site, layer)] = seed_signs(n, seed_index, site, layer)
            if method != 'hadamard':
                self.factors[(site, layer)] = act.RotationFactor.load(factor_path(args, model, method, site, layer), torch.device('cuda'))

    def apply(self, site, layer, value):
        signs = self.signs[(site, layer)]
        if self.method == 'hadamard':
            return act.full_hadamard_rows(value.float(), signs)
        return self.factors[(site, layer)].apply(value, signs)


def fp32_nll(model, batch):
    with torch.inference_mode():
        output = model(input_ids=batch.cuda(), use_cache=False)
        logits = output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1])
        labels = batch[:, 1:].to(logits.device).reshape(-1)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        assert loss.dtype == torch.float32
        value = float(loss)
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite fp32 NLL')
        return value


def evaluate(args, model_key, methods=(*VARIANTS, 'hadamard')):
    _, tokens, signature = audit_inputs(args, model_key)
    _, results = roots(args, model_key)
    output = results / 'e27_per_sequence.partial.csv'
    rows = csv_rows(output)
    # A requeue can arrive during D evaluation; finish every checkpointed method.
    methods = tuple(dict.fromkeys([*methods, *(r['method'] for r in rows if r['method'] != 'bf16')]))
    done = {(r['site'], r['method'], int(r['seed']), int(r['sequence'])) for r in rows}
    model = base.load_model(act.MODEL_IDS[model_key], args.workdir)
    def measure(site, method, seed, fold_error):
        for seq in range(NEVAL):
            key = (site, method, seed, seq)
            if key in done:
                continue
            loss = fp32_nll(model, tokens[seq:seq + 1])
            rows.append({'model': model_key, 'model_id': act.MODEL_IDS[model_key], 'site': site,
                         'method': method, 'seed': seed, 'sequence': seq, 'nll': loss,
                         'tokens_scored': LENGTH - 1, 'nll_dtype': 'fp32',
                         'weight_fold_max_relative_error': fold_error, 'source': 'E27 matched hardware evaluation',
                         'protocol_sha256': signature})
            done.add(key)
            if (seq + 1) % 8 == 0:
                write_csv(output, rows)
                LOG.info('E27 PPL %s %s %s seed=%d sequence=%d/%d', model_key, method, site, seed, seq + 1, NEVAL)
        write_csv(output, rows)
    measure('none', 'bf16', -1, 0.)
    manager = act.WeightManager(model)
    for method in methods:
        for si in range(3):
            if all((condition, method, SEED + si, seq) in done for condition in CONDITIONS for seq in range(NEVAL)):
                continue
            rotation = Rotations(args, model_key, method, si)
            manager.restore('qkv'); manager.restore('down')
            qerror = manager.rotate('qkv', rotation, 512)
            derror = 0.
            for condition in CONDITIONS:
                if condition == 'both':
                    derror = manager.rotate('down', rotation, 512)
                if condition == 'down_only':
                    manager.restore('qkv')
                error = max(qerror if condition != 'down_only' else 0, derror)
                if error > .02:
                    raise RuntimeError(f'Weight fold error {error}; refusing PPL evaluation')
                hooks = act.ActivationQuantHooks(model, rotation, *act._condition_flags(condition))
                hooks.install()
                try:
                    measure(condition, method, SEED + si, error)
                finally:
                    hooks.close()
            del rotation
            gc.collect(); torch.cuda.empty_cache()
    manager.restore('qkv'); manager.restore('down')
    del manager, model
    gc.collect(); torch.cuda.empty_cache()
    write_csv(results / 'e27_per_sequence.csv', rows)
    summary = summarize(rows)
    write_csv(results / 'e27_summary.csv', summary)
    return summary


def summarize(rows):
    output = []
    methods = sorted({r['method'] for r in rows if r['method'] != 'bf16'})
    ppls = {}
    for site in CONDITIONS:
        for method in methods:
            for seed in range(SEED, SEED + 3):
                part = [r for r in rows if r['site'] == site and r['method'] == method and int(r['seed']) == seed]
                if len(part) != NEVAL or {int(r['sequence']) for r in part} != set(range(NEVAL)):
                    raise RuntimeError(f'Incomplete/duplicated PPL: {site} {method} {seed}')
                ppls[(site, method, seed)] = math.exp(float(np.mean([float(r['nll']) for r in part])))
        for method in methods:
            values = np.array([ppls[(site, method, s)] for s in range(SEED, SEED + 3)])
            delta = values - np.array([ppls[(site, 'A_full', s)] for s in range(SEED, SEED + 3)])
            half = act.TCRIT_DF2_90 * float(delta.std(ddof=1)) / math.sqrt(3)
            output.append({'model': rows[0]['model'], 'site': site, 'method': method, 'seeds': 3,
                           'mean_ppl': float(values.mean()), 'seed_ppl_std': float(values.std(ddof=1)),
                           'paired_ppl_delta_vs_A': float(delta.mean()),
                           'paired_90ci_low_vs_A': float(delta.mean() - half),
                           'paired_90ci_high_vs_A': float(delta.mean() + half),
                           'seed_ppls': ';'.join(format(v, '.12g') for v in values),
                           'seed_values': ';'.join(str(s) for s in range(SEED, SEED + 3)),
                           'eval_sequences': NEVAL, 'tokens_scored_per_sequence': LENGTH - 1})
    return output


def trigger_d(summary):
    return [r for r in summary if r['method'].startswith('B_') and float(r['paired_ppl_delta_vs_A']) < 0]


def publish(args, model):
    _, results = roots(args, model)
    out = args.repo / 'results' / model
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(results.glob('e27_*.csv')) + sorted(results.glob('e27_*.json')) + [results / 'E27_DONE.json']:
        if path.exists() and '.partial.' not in path.name:
            shutil.copy2(path, out / path.name)


def finalize(args, model):
    folder, results = roots(args, model)
    summary = summarize(csv_rows(results / 'e27_per_sequence.csv'))
    needs_d = bool(trigger_d(summary))
    methods = [*VARIANTS, 'hadamard'] + (['D_range_direct'] if needs_d else [])
    assert {r['method'] for r in summary} == set(methods)
    diagnostics_rows = csv_rows(results / 'e27_f_and_angles.csv')
    expected = {(site, layer, method, seed) for site, layer, _ in layer_keys(args, model)
                for method in methods for seed in range(SEED, SEED + 3)}
    actual = {(r['site'], int(r['layer']), r['method'], int(r['seed'])) for r in diagnostics_rows}
    if actual != expected or len(actual) != len(diagnostics_rows):
        raise RuntimeError('Incomplete or duplicate per-layer diagnostics')
    factor_audit = []
    for site, layer, n in layer_keys(args, model):
        for method in methods:
            if method in ('A_full', 'hadamard'):
                continue
            path = factor_path(args, model, method, site, layer)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            original = torch.load(factor_path(args, model, 'A_full', site, layer), map_location='cpu', weights_only=True)
            assert torch.equal(payload['source_order'], original['source_order'])
            assert torch.equal(payload['target_order'], original['target_order'])
            factor_audit.append({'model': model, 'method': method, 'site': site, 'layer': layer,
                                 'k': n // GROUP, 'anchor_error': payload['anchor_error'],
                                 'selected_numerical_rank': payload.get('selected_numerical_rank', -1),
                                 'null_completion_directions': payload.get('null_completion_directions', 0),
                                 'permutation_identical_to_A': True, 'factor_sha256': digest(path)})
    write_csv(results / 'e27_factor_audit.csv', factor_audit)
    write_csv(results / 'e27_summary.csv', summary)
    write_csv(results / 'e27_law_fit.csv', law_fits(diagnostics_rows))
    old = csv_rows(args.repo / 'results' / model / 'e5_per_sequence.csv')
    comparison = []
    fresh_rows = csv_rows(results / 'e27_per_sequence.csv')
    for condition in CONDITIONS:
        for method, old_name in [('A_full', 'nar'), ('hadamard', 'hadamard')]:
            for seed in range(SEED, SEED + 3):
                prior = [float(r['nll']) for r in old if r['site'] == condition and r['method'] == old_name and int(r['seed']) == seed]
                fresh = [float(r['nll']) for r in fresh_rows if r['site'] == condition and r['method'] == method and int(r['seed']) == seed]
                if len(prior) != NEVAL:
                    raise RuntimeError('Incomplete original E5 audit reference')
                p0, p1 = math.exp(float(np.mean(prior))), math.exp(float(np.mean(fresh)))
                comparison.append({'model': model, 'site': condition, 'method': method, 'seed': seed, 'E5_ppl': p0, 'E27_ppl': p1, 'hardware_replay_ppl_delta': p1 - p0})
    write_csv(results / 'e27_baseline_replay.csv', comparison)
    pre, signature = protocol(args)
    base.atomic_json(results / 'E27_DONE.json', {
        'experiment': 'E27', 'model': model, 'completed_at_utc': utc(), 'protocol_sha256': signature,
        'preregistered_at_utc': pre['preregistered_at_utc'], 'preregistration': 'experiments/e27_preregistration.json',
        'methods': methods, 'D_triggered': needs_d, 'D_trigger_rows': trigger_d(summary),
        'capture': read_json(folder / 'CAPTURE_DONE.json'), 'source_sha256': digest(Path(__file__)),
        'paired_CI': pre['paired_ci'], 'no_tuning': True, 'A_factors_reused_unchanged': pre['assets'][model]['A_factors_sha256'],
        'result_sha256': {p.name: digest(p) for p in results.glob('e27_*.csv') if '.partial.' not in p.name},
        'hardware': base.hardware_info(),
    })
    publish(args, model)


def update_report(args):
    lines = ['<!-- E27_RESULTS_BEGIN -->', '\n### E27 execution and results\n']
    for model in protocol(args)[0]['models']:
        path = args.repo / 'results' / model
        if not (path / 'E27_DONE.json').exists():
            lines.append(f'**{model}:** pending completion; no conclusion yet.\n')
            continue
        done = read_json(path / 'E27_DONE.json')
        lines += [f'**{model}: complete.** D triggered: {done["D_triggered"]}.\n',
                  '| Site | Variant | Mean PPL | Paired delta vs A [90% CI] |', '|---|---|---:|---:|']
        for row in csv_rows(path / 'e27_summary.csv'):
            lines.append(f'| {row["site"]} | {row["method"]} | {float(row["mean_ppl"]):.5f} | {float(row["paired_ppl_delta_vs_A"]):+.5f} [{float(row["paired_90ci_low_vs_A"]):+.5f}, {float(row["paired_90ci_high_vs_A"]):+.5f}] |')
        rows = csv_rows(path / 'e27_f_and_angles.csv')
        lines += ['', '| Site | Variant | Mean f | Mean range/Hadamard | Mean NMSE | Mean top-8 angle vs A (deg) |', '|---|---|---:|---:|---:|---:|']
        for site in act.SITES:
            for method in done['methods']:
                part = [r for r in rows if r['site'] == site and r['method'] == method]
                stats = [np.mean([float(r[k]) for r in part]) for k in ['f', 'range_over_paired_hadamard', 'relative_quantization_error_nmse', 'angles_mean_degrees_vs_A_top8']]
                lines.append('| ' + ' | '.join([site, method] + [f'{x:.5g}' for x in stats]) + ' |')
        b_rows = [r for r in csv_rows(path / 'e27_summary.csv') if r['site'] == 'both' and r['method'].startswith('B_')]
        for r in b_rows:
            delta, lo, hi = (float(r[k]) for k in ['paired_ppl_delta_vs_A', 'paired_90ci_low_vs_A', 'paired_90ci_high_vs_A'])
            evidence = 'B lower than A with a paired interval below zero' if hi < 0 else 'A lower than B with a paired interval above zero' if lo > 0 else 'paired interval overlaps zero; inconclusive ordering'
            lines.append(f'\n{r["method"]}, both sites: {evidence}.')
        rank_rows = [r for r in csv_rows(path / 'e27_factor_audit.csv') if r['method'].startswith('B_') and int(r['selected_numerical_rank']) < 8]
        if rank_rows:
            lines.append(f'\n{len(rank_rows)} B layer/site estimates have fewer than eight identified directions. Their required top-8 angle rows include the preregistered null-space completion and must not be interpreted as eight data-identified directions.')
        lines.append('\nLayer/seed rows and all eight angles are retained in the CSV. Repeated seeds are not independent layer replicates.\n')
    lines.append('<!-- E27_RESULTS_END -->')
    path = args.repo / 'report.md'
    text = path.read_text()
    begin, end = '<!-- E27_RESULTS_BEGIN -->', '<!-- E27_RESULTS_END -->'
    replacement = '\n'.join(lines)
    standalone = args.repo / 'results/e27_report.md'
    standalone.write_text('# E27 — which tokens define the subspace\n\n' + replacement + '\n')
    if begin in text:
        lo, hi = text.index(begin), text.index(end) + len(end)
        text = text[:lo] + replacement + text[hi:]
    else:
        text += '\n\n' + replacement + '\n'
    text = text.replace('**Execution status:** preregistered; implementation and validation in progress. No E27 result or H-A/H-B conclusion is available yet.',
                        '**Execution status:** see the completion-gated E27 tables below. Pending stages are not results.')
    path.write_text(text)



def replay_audit(args, model):
    folder, results = roots(args, model)
    path = results / 'e27_capture_replay.csv'
    existing = csv_rows(path)
    completed = {(r['site'], int(r['layer'])) for r in existing}
    original = csv_rows(args.repo / 'results' / model / 'e1c_per_layer.csv')
    for site, layer, n in layer_keys(args, model):
        if (site, layer) in completed:
            continue
        x = evaluation_rows(args, model, site, layer, n)
        raw_range = float((x.reshape(-1, n // GROUP, GROUP).amax(-1) - x.reshape(-1, n // GROUP, GROUP).amin(-1)).double().mean())
        source_site = 'q_input' if site == 'qkv' else 'down_input'
        ref = [r for r in original if r['site'] == source_site and int(r['layer']) == layer and r['method'] == 'identity']
        old = float(ref[0]['mean_group_range']) if ref else math.nan
        drift = raw_range / old - 1 if ref else math.nan
        if ref and abs(drift) > .005:
            raise RuntimeError(f'E1c raw range replay differs by >0.5%: {model} {site} {layer} {drift}')
        existing.append({'model': model, 'site': site, 'layer': layer, 'evaluation_tokens': len(x),
                         'recaptured_raw_mean_group_range': raw_range, 'original_E1c_raw_mean_group_range': old,
                         'relative_range_replay_drift': drift,
                         'reference_status': 'original E1c identity rows' if ref else 'no original E1c dump or per-layer reference for this model',
                         'provenance': 'same frozen token IDs and E1c row positions; freshly replayed bf16 values'})
        write_csv(path, existing)
        del x
    if model == 'llama32_3b':
        import numpy as np
        hero = np.load(args.repo / 'figures/fig1_source_arrays.npz')
        meta = read_json(args.repo / 'figures/fig1_metadata.json')
        n = read_json(folder / 'CAPTURE_DONE.json')['dimensions']['down']
        x = evaluation_rows(args, model, 'down', meta['layer'], n)
        seq, token = meta['hero']['sequence_index'], meta['hero']['token_position']
        assert token % STRIDE == 0
        start = meta['channel_windows']['raw'][0]
        old = hero['raw_magnitude'][int(np.flatnonzero(hero['token_axis'] == token)[0])]
        fresh = x[seq * (LENGTH // STRIDE) + token // STRIDE, start:start + len(old)].abs().cpu().numpy()
        relative = float(np.linalg.norm(fresh - old) / np.linalg.norm(old))
        if relative > .01:
            raise RuntimeError(f'Frozen Figure 1 raw hero replay drift exceeds 1%: {relative}')
        base.atomic_json(results / 'e27_hero_replay.json', {'relative_l2_drift': relative,
            'exactly_equal_fraction': float(np.mean(fresh == old)), 'sequence': seq, 'token': token,
            'layer': meta['layer'], 'channel_start': start, 'channels': len(old)})


def law_fits(rows):
    fits = []
    for site in act.SITES:
        methods = sorted({r['method'] for r in rows if r['site'] == site and r['method'] != 'hadamard'})
        for method in methods + ['pooled']:
            selected = [r for r in rows if r['site'] == site and r['method'] != 'hadamard' and (method == 'pooled' or r['method'] == method)]
            pairs = sorted({(int(r['layer']), r['method']) for r in selected})
            points = []
            for layer, name in pairs:
                group = [r for r in selected if int(r['layer']) == layer and r['method'] == name]
                points.append([np.mean([float(r['sqrt_one_minus_f']) for r in group]),
                               np.mean([float(r['range_over_paired_hadamard']) for r in group])])
            x, y = np.asarray(points).T
            design = np.column_stack([np.ones_like(x), x])
            coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coefficient
            variance = float(np.square(y - y.mean()).sum())
            fits.append({'model': rows[0]['model'], 'site': site, 'method': method, 'layer_method_points': len(x),
                         'aggregation': 'average three paired rotation seeds per layer/method before descriptive OLS',
                         'intercept': float(coefficient[0]), 'slope': float(coefficient[1]),
                         'r_squared': 1 - float(np.square(residual).sum()) / variance if variance > 0 else math.nan,
                         'identity_rmse': float(np.sqrt(np.mean((y - x) ** 2)))})
    return fits


def gpu_smoke(args):
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    x = torch.randn((128, 512), generator=generator).cuda()
    vectors, _ = estimate_basis(x, 4, SEED, exact=True)
    c_vectors, _ = estimate_basis(x, 4, SEED, exact=False)
    factor = act.factor_from_vectors(vectors, x, GROUP)
    signs = seed_signs(512, 0, 'qkv', 0)
    rotated = factor.apply(x, signs)
    expected = (x @ vectors).square().sum() / x.square().sum()
    actual = GROUP * rotated.reshape(-1, 4, GROUP).mean(-1).square().sum() / rotated.square().sum()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    weights = torch.randn((16, 512), generator=generator).cuda()
    torch.testing.assert_close(rotated @ factor.apply(weights, signs).T, x @ weights.T, rtol=1e-4, atol=5e-5)
    train, test, _ = audit_inputs(args, 'llama32_3b')
    model = base.load_model(act.MODEL_IDS['llama32_3b'], args.workdir)
    batch = test[:1, :128]
    measured = fp32_nll(model, batch)
    with torch.inference_mode():
        native = float(model(input_ids=batch.cuda(), labels=batch.cuda(), use_cache=False).loss)
    if abs(measured - native) > 2e-6:
        raise RuntimeError(f'Explicit fp32 NLL differs from E5 model loss: {measured} {native}')
    base.atomic_json(args.workdir / 'results/e27_gpu_smoke.json', {'status': 'PASS', 'at_utc': utc(),
        'explicit_fp32_nll': measured, 'native_E5_loss': native, 'difference': measured - native,
        'projection_capture_and_exact_transpose_fold': 'PASS', 'hardware': base.hardware_info()})
    del model, x, vectors, c_vectors, factor
    gc.collect(); torch.cuda.empty_cache()


def run(args, model):
    _, results = roots(args, model)
    if (results / 'E27_DONE.json').exists():
        publish(args, model)
        update_report(args)
        return
    capture(args, model)
    replay_audit(args, model)
    build(args, model)
    diagnostics(args, model)
    summary = evaluate(args, model)
    publish(args, model)
    if trigger_d(summary):
        base.atomic_json(results / 'e27_D_TRIGGER.json', {'observed_at_utc': utc(), 'trigger_rows': trigger_d(summary)})
        LOG.info('E27 %s: B mean PPL below A; running preregistered D', model)
        build(args, model, ['D_range_direct'])
        diagnostics(args, model, ['D_range_direct'])
        evaluate(args, model, ['D_range_direct'])
    finalize(args, model)
    update_report(args)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--models', nargs='+', choices=['llama32_3b', 'llama31_8b'], default=['llama32_3b', 'llama31_8b'])
    p.add_argument('stage', choices=['audit', 'smoke', 'capture', 'build', 'diagnostics', 'evaluate', 'finalize', 'report', 'run'])
    return p


def main():
    args = parser().parse_args()
    args.workdir, args.repo = args.workdir.resolve(), args.repo.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base.seed_everything(SEED)
    if args.stage == 'report':
        update_report(args)
        return
    if args.stage != 'audit' and not torch.cuda.is_available():
        raise RuntimeError('E27 requires an allocated single CUDA GPU')
    base.setup_logging(args.workdir, 'e27-direction-selection')
    if args.stage == 'smoke':
        gpu_smoke(args)
        return
    for model in args.models:
        if args.stage == 'audit':
            audit_inputs(args, model)
            LOG.info('E27 frozen assets PASS: %s', model)
        else:
            globals()[args.stage](args, model)


if __name__ == '__main__':
    main()
