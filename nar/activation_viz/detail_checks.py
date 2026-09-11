"""Audit real local windows and immutable experiment artifacts; no recapture."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .detail_plot import (CONFIG, METHODS, digest, load_local, local_values,
                          reference_audit, write_json)


def snapshot(root):
    root = Path(root)
    paths = [root/n for n in ('metrics_per_sample.csv', 'metrics_summary.csv',
             'activation_inventory.json', 'inputs.json', 'validation_report.json',
             'numerical_edge_checks.json', 'measured_summary.md')]
    paths += list((root/'exact_ecdf').rglob('*'))
    paths += [p for p in (root/'figures').rglob('*') if 'detail' not in p.parts]
    result = {str(p.relative_to(root)): digest(p) for p in paths if p.is_file()}
    manifest = json.loads((root/'run_manifest.json').read_text())
    for key in ('display', 'display_revision_history', 'render_commands'):
        manifest.pop(key, None)
    result['scientific_run_manifest'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return result


def numerical_checks():
    # Synthetic arrays are only coordinate/order tests and are never exported.
    y = torch.arange(256*640, dtype=torch.float32).reshape(256, 640) - 50
    raw = local_values(y, 'raw')
    residual = local_values(y, 'residual')
    assert raw[7, 499] == abs(float(y[7, 499]))
    assert residual[7, 499] == abs(float(y[7, 499] - y[7, 384:512].mean()))
    assert not np.array_equal(residual, local_values(y.abs(), 'residual'))
    before = y.clone(); local_values(y, 'residual'); assert torch.equal(y, before)
    zero = local_values(torch.zeros((256, 640)), 'raw')
    assert zero.shape == (128, 512) and not zero.any()
    x, t = np.meshgrid(np.arange(512), np.arange(128), indexing='xy')
    for token, channel in [(0, 0), (0, 511), (127, 0), (127, 511), (7, 499)]:
        assert (x[token, channel], t[token, channel]) == (channel, token)
    return {'passed': True, 'synthetic_role': 'unit tests only; never paper figures',
            'signed_centering_before_abs': True, 'array_not_modified': True,
            'integer_xy_not_transposed': True, 'all_zero_window': True}


def run(source, destination, baseline=None, all_shards=True):
    torch.set_num_threads(1)
    source, destination = Path(source), Path(destination)
    result = {'unit_checks': numerical_checks(), 'reference': reference_audit(str(source), str(destination))}
    baseline = Path(baseline) if baseline else destination/'qa/detail_immutable_before.json'
    if baseline.exists():
        before = json.loads(baseline.read_text())
        after = snapshot(destination)
        assert before == after, 'Metrics, validation, ECDFs or preserved overviews changed'
        result['immutable_artifacts'] = {'passed': True, 'files': len(before), 'sha256': before}
    inventory = json.loads((destination/'activation_inventory.json').read_text())
    selected = inventory if all_shards else [r for r in inventory if r['sample'] == 0]
    def check_shard(r):
        actual = digest(source/r['file'])
        assert actual == r['sha256'], r['file']
        return {'file': r['file'], 'sha256': actual}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(check_shard, selected))
    result['raw_shards'] = {'passed': True, 'count': len(hashes), 'records': hashes}
    manifest = json.loads((destination/'run_manifest.json').read_text())
    work = source.parents[2]
    factors = {r['path']: r['sha256'] for spec in manifest['rotation_factors'].values() for r in spec['files']}
    for path, expected in factors.items():
        assert digest(work/path) == expected, path
    result['rotation_factors'] = {'passed': True, 'count': len(factors), 'sha256': factors}
    panel_count = 0
    for path in sorted((destination/'figures').rglob('*.geometry.json')):
        if 'detail' not in path.parts: continue
        g = json.loads(path.read_text())
        assert g['config']['revision'] == CONFIG['revision']
        for panel in g['panels']:
            z, record = load_local(str(source), str(destination), panel['display_mode'],
                panel['method'], panel['layer'], panel['site'], panel['quantity'])
            assert panel['local_array_sha256'] == record['local_array_sha256']
            assert panel['shape'] == [128, 512] and panel['surface_cells'] == 127*511
            assert panel['surface_rcount'] == 128 and panel['surface_ccount'] == 512
            assert panel['draw_surface'] and not panel['draw_sidewalls'] and not panel['draw_height_columns']
            assert not panel['zero_padding'] and not panel['pooling']
            assert panel['vertical_height_segments'] == panel['sidewall_triangles'] == 0
            for p in panel['debug_points']:
                assert p['z'] == float(z[p['token'], p['channel']])
            layer = str(panel['layer']); limit = g['scales']['z_limits_by_layer'][layer]
            assert panel['z_limit'] == panel['norm']['vmax'] == limit
            assert panel['norm']['gamma'] == g['scales']['gamma']
            maxima = [load_local(str(source),str(destination),panel['display_mode'],m,panel['layer'],panel['site'],panel['quantity'])[0].max() for m in g['scales']['shared_across_methods']]
            maximum = float(max(maxima)); assert limit == (maximum*1.03 if maximum else 1.)
            panel_count += 1
    for mode in ('paired_local', 'end_to_end'):
        for site in ('q_proj', 'down_proj'):
            for quantity in ('raw', 'residual'):
                d = destination/'figures'/mode/site/quantity/'detail'
                primary = json.loads((d/'matrix.geometry.json').read_text())
                linear = json.loads((d/'matrix_linear.geometry.json').read_text())
                assert [p['method'] for p in primary['panels'][:4]] == METHODS
                assert len(primary['panels']) == 16
                for p, q in zip(primary['panels'], linear['panels']):
                    assert all(p[k] == q[k] for k in ('source_sha256','local_array_sha256','z_limit','camera'))
                assert digest(d/'matrix.pdf') != digest(d.parent/'overview/matrix.pdf')
    result.update(passed=True, local_panel_records_checked=panel_count,
                  matrix_linear_same_data_camera_limits=True, overview_detail_independent=True)
    write_json(destination/'qa/detail_integrity.json', result)
    print('DETAIL INTEGRITY PASS', len(hashes), 'shards,', len(factors), 'factors,', panel_count, 'panel records')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination')
    p.add_argument('--snapshot', action='store_true');a=p.parse_args()
    if a.snapshot: write_json(Path(a.destination)/'qa/detail_immutable_before.json', snapshot(a.destination))
    else: run(a.source,a.destination)
