"""Fixed, measured 128 x 512 local surfaces; no recapture or quantizer changes."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps, colors, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d

from .plot import LAYERS, METHODS, LABELS, number, plt, style
from .qa.panel_alignment import require_matplotlib_panel_alignment

CONFIG = {
    'revision': 'local-upper-surface-v1', 'sample': 0,
    'tokens': [0, 128], 'channels': [0, 512], 'group_size': 128,
    'group_boundaries': [127.5, 255.5, 383.5],
    'shape': [128, 512], 'data_stride': [1, 1], 'pooling': False,
    'draw_surface': True, 'draw_height_columns': False, 'draw_sidewalls': False,
    'zero_padding': False, 'height_transform': 'absolute value only; linear z axis',
    'raw': 'abs(signed_Y[0:128, 0:512])',
    'residual': 'abs((Y - per-token contiguous-g128 signed mean)[0:128, 0:512])',
    'residual_centering_domain': 'complete signed Y, before crop and abs',
    'camera': {'projection': 'ortho', 'elev': 28, 'azim': -55,
               'box_aspect': [2.0, 1.2, 1.0]},
    'cmap': 'viridis', 'norm': 'PowerNorm', 'gamma': 0.5,
    'vmin': 0, 'vmax_rule': '1.03 * maximum across displayed methods per column',
    'all_zero_policy': 'record all_zero=true and use display-only vmax=1',
    'linear_control': 'Normalize with identical data, camera and z limits',
    'shade': False, 'surface_rcount': 128, 'surface_ccount': 512,
    'surface_dpi': 600, 'formats': ['pdf', 'png', 'svg'],
    'text_axes': 'vector', 'font_size_pt': 7,
    'reference_source_mode': 'paired_local',
    'reference_precision': 'BF16 checkpoint loaded and norm-fused in FP32',
    'reference_label': 'Unrotated reference (norm-fused FP32)',
    'end_to_end_rows': 'reference is unquantized; other rows are existing E22 W4A4KV4 QDQ forwards',
    'statistical_unit': 'fixed sample 0; descriptive surface, no uncertainty estimate',
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


@lru_cache(maxsize=4)
def reference_audit(source, destination):
    """Validate the run-wide input identity and every sample-0 reference pairing."""
    source, destination = Path(source), Path(destination)
    sm = json.loads((source / 'run_manifest.json').read_text())
    dm = json.loads((destination / 'run_manifest.json').read_text())
    si = json.loads((source / 'inputs.json').read_text())
    di = json.loads((destination / 'inputs.json').read_text())
    assert si == di, 'Reference and publication input descriptions differ'
    ids = torch.load(source / 'inputs.pt', weights_only=True, map_location='cpu')
    assert torch.equal(ids['input_ids'], torch.tensor(si['input_ids']))
    assert torch.equal(ids['sample_indices'], torch.tensor(si['sample_indices']))
    assert torch.all(ids['attention_mask'] == 1)
    assert tensor_digest(ids['input_ids']) == si['input_ids_sha256'] == sm['input_ids_sha256'] == dm['input_ids_sha256']
    for key in ('base_revision', 'model_id', 'layers_zero_based', 'sites', 'reference', 'rotation_factors'):
        assert sm[key] == dm[key], key
    assert 'FP32' in sm['reference'] and 'norm-fused' in sm['reference']
    assert digest(source / 'capture_site_report.md') == digest(destination / 'capture_site_report.md')
    inventory = json.loads((destination / 'activation_inventory.json').read_text())
    assert inventory == json.loads((source / 'activation_inventory.json').read_text())
    entries = {r['file']: r for r in inventory}
    pairs = []
    for layer in LAYERS:
        for site in ('q_proj', 'down_proj'):
            records = [entries[f'activations/paired_local/{m}/s00_l{layer:02d}_{site}.pt'] for m in METHODS]
            canonical = records[0]['tensor_sha256']
            assert all(r['canonical_tensor_sha256'] == canonical for r in records)
            for mode in ('paired_local', 'end_to_end'):
                for method in (METHODS if mode == 'paired_local' else METHODS[1:]):
                    r = entries[f'activations/{mode}/{method}/s00_l{layer:02d}_{site}.pt']
                    assert (r['sample'], r['layer'], r['site'], r['dtype']) == (0, layer, site, 'torch.float32')
                    assert r['shape'] == [2048, 12288 if site == 'down_proj' else 4096]
            pairs.append({'layer': layer, 'site': site, 'reference': records[0]['file'],
                          'canonical_tensor_sha256': canonical})
    return {'passed': True, 'sample': 0, 'sample_index': si['sample_indices'][0],
            'input_ids_sha256': si['input_ids_sha256'],
            'sample0_input_ids_sha256': tensor_digest(ids['input_ids'][0]),
            'decoded_text_sha256': si['decoded_text_sha256'],
            'reference': sm['reference'], 'capture_site_report_sha256': digest(source / 'capture_site_report.md'),
            'evidence': 'same saved inputs.pt and sample-index loop in capture.py for both modes; inventory matches capture run',
            'end_to_end_canonical_identity_claimed': False, 'pairs': pairs}


def local_values(y, quantity):
    """Center the complete signed tensor before taking absolute local heights."""
    if quantity == 'residual':
        groups = y.reshape(y.shape[0], -1, 128)
        values = (groups - groups.mean(-1, keepdim=True)).reshape_as(y)
    elif quantity == 'raw':
        values = y
    else:
        raise ValueError(quantity)
    z = values[:128, :512].abs().contiguous().numpy()
    assert z.shape == (128, 512) and np.isfinite(z).all()
    return z


@lru_cache(maxsize=128)
def load_local(source, destination, mode, method, layer, site, quantity):
    reference_audit(str(source), str(destination))
    source_mode = 'paired_local' if method == 'unrotated' else mode
    rel = f'activations/{source_mode}/{method}/s00_l{layer:02d}_{site}.pt'
    path = Path(source) / rel
    inventory = json.loads((Path(destination) / 'activation_inventory.json').read_text())
    entry = next(r for r in inventory if r['file'] == rel)
    actual_hash = digest(path)
    assert actual_hash == entry['sha256'], rel
    y = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    assert y.dtype == torch.float32 and list(y.shape) == entry['shape'] and torch.isfinite(y).all()
    assert tensor_digest(y) == entry['tensor_sha256']
    z = local_values(y, quantity)
    # Independent per-group expression checks the crop, sign/abs order and all values.
    expected = y[:128, :512].clone()
    if quantity == 'residual':
        for start in range(0, 512, 128):
            expected[:, start:start+128] -= y[:128, start:start+128].mean(-1, keepdim=True)
    assert np.array_equal(z, expected.abs().numpy())
    record = {'source': rel, 'source_sha256': actual_hash,
              'source_tensor_sha256': entry['tensor_sha256'],
              'source_shape': entry['shape'], 'local_array_sha256': hashlib.sha256(z.tobytes()).hexdigest(),
              'method': method, 'layer': layer, 'site': site, 'quantity': quantity,
              'display_mode': mode, 'source_mode': source_mode, 'sample': 0,
              'minimum': float(z.min()), 'maximum': float(z.max()),
              'canonical_tensor_sha256': entry['canonical_tensor_sha256']}
    return z, record


def setup_axes(ax, limit, labels=True):
    ax.set_proj_type('ortho')
    ax.view_init(elev=28, azim=-55)
    ax.set_box_aspect((2.0, 1.2, 1.0))
    ax.set_xlim(0, 511); ax.set_ylim(0, 127); ax.set_zlim(0, limit)
    ax.set_xmargin(0); ax.set_ymargin(0); ax.set_zmargin(0)
    ax.set_xticks([0, 128, 256, 384, 511])
    ax.set_yticks([0, 32, 64, 96, 127])
    ax.set_zticks([0, limit / 2, limit])
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(number))
    ax.tick_params(axis='both', labelsize=7, pad=0)
    ax.tick_params(axis='z', labelsize=7, pad=1)
    if labels:
        ax.set_xlabel('Channel index', fontsize=7, labelpad=9)
        ax.set_ylabel('Token index', fontsize=7, labelpad=9)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.985, .985, .985, 1))
        axis.line.set_color('#a0a0a0'); axis.line.set_linewidth(.4)
        axis._axinfo['grid']['linewidth'] = .25
        axis._axinfo['grid']['color'] = (.86, .86, .86, 1)
    for boundary in CONFIG['group_boundaries']:
        ax.plot([boundary, boundary], [0, 127], [0, 0], color='.45',
                linewidth=.45, linestyle=(0, (3, 3)))


def surface(ax, z, norm):
    channel_idx = np.arange(512)
    token_idx = np.arange(128)
    x, y = np.meshgrid(channel_idx, token_idx, indexing='xy')
    assert x.shape == y.shape == z.shape == (128, 512)
    artist = ax.plot_surface(x, y, z, rcount=z.shape[0], ccount=z.shape[1],
                            cmap=colormaps['viridis'], norm=norm, shade=False,
                            linewidth=0, antialiased=False, rasterized=True)
    return artist


def point_records(z):
    points = [(0, 0), (511, 0), (0, 127), (511, 127)]
    peaks = np.argsort(z.ravel(), kind='stable')[-3:][::-1]
    points += [(int(i % 512), int(i // 512)) for i in peaks]
    return [{'id': f'P{i}', 'channel': x, 'token': y, 'z': float(z[y, x])}
            for i, (x, y) in enumerate(points)]


def export(fig, path, records, scales, colorbars):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(path) + '.alignment.json',
                                       strict=True, exclude_axes=colorbars)
    fig.savefig(str(path) + '.pdf', dpi=600)
    fig.savefig(str(path) + '.svg', dpi=600)
    fig.savefig(str(path) + '.png', dpi=600)
    write_json(path.with_suffix('.geometry.json'), {
        'config': CONFIG, 'panels': records, 'scales': scales,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'plot_source_sha256': digest(__file__),
        'output_paths': [str(path.with_suffix('.' + ext)) for ext in ('pdf', 'png', 'svg')],
        'figure_inches': list(fig.get_size_inches()),
        'color_semantics': 'Matplotlib surface face mean height mapped through the declared norm; vertex heights unchanged',
    })
    write_json(path.with_suffix('.scales.json'), scales)
    plt.close(fig)


def matrix(source, destination, mode, site, quantity, methods=METHODS, layers=LAYERS,
           name='matrix', linear=False, scale_methods=None, debug=False, directory=None):
    torch.set_num_threads(1)
    style()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 7, 'axes.labelsize': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7})
    candidates = list(scale_methods or methods)
    rows, cols = len(methods), len(layers)
    width, height = 1.45 + 2.6 * cols + (1.6 if debug else 0), .95 + 2.35 * rows
    fig = plt.figure(figsize=(width, height), dpi=600)
    grid = fig.add_gridspec(rows, cols, left=1.15 / width, right=1 - (.5 + (1.6 if debug else 0)) / width,
                           bottom=.85 / height, top=1 - .6 / height,
                           hspace=.14, wspace=.16)
    title = 'Pre-quantization activation magnitude' if quantity == 'raw' else 'Group-centered residual magnitude'
    fig.text(.5, 1 - .1 / height, title, fontsize=9, ha='center', va='top')
    mapping = 'Linear heights; linear color mapping.' if linear else 'Linear heights; square-root color mapping.'
    zoom = name == 'rotated_only_zoom'
    fig.text(.5, .06 / height, 'Sample 0 · tokens 0–127 · channels 0–511 · g128.\n' + mapping,
             ha='center', va='bottom', fontsize=7)
    if zoom:
        fig.text(.5, 1 - .28 / height, 'Rotated-only zoom: shared scale differs from the four-row matrix.',
                 ha='center', va='top', fontsize=7)
    records, axes, colorbars, limits, zero_flags = [], {}, [], {}, {}
    for col, layer in enumerate(layers):
        data = {m: load_local(str(source), str(destination), mode, m, layer, site, quantity) for m in candidates}
        maximum = max(float(v[0].max()) for v in data.values())
        limit = 1.03 * maximum if maximum else 1.0
        limits[str(layer)] = limit; zero_flags[str(layer)] = maximum == 0
        norm = colors.Normalize(0, limit) if linear else colors.PowerNorm(.5, vmin=0, vmax=limit)
        for row, method in enumerate(methods):
            z, original = data[method] if method in data else load_local(str(source), str(destination), mode, method, layer, site, quantity)
            ax = fig.add_subplot(grid[row, col], projection='3d')
            axes[row, col] = ax
            setup_axes(ax, limit)
            surface(ax, z, norm)
            if row == 0:
                ax.set_title(f'Block {layer + 1}', fontsize=8, pad=5)
            if col == 0:
                label = 'Unrotated reference\n(norm-fused FP32)' if method == 'unrotated' else LABELS[method].replace(' (', '\n(')
                position = ax.get_position()
                fig.text(.06 / width, position.y0 + position.height / 2, label,
                         ha='left', va='center', fontsize=8)
            points = point_records(z)
            if debug:
                marker_colors = ['#000000', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#663399']
                for point, marker_color in zip(points, marker_colors):
                    ax.scatter(point['channel'], point['token'], point['z'], color=marker_color, s=14, depthshade=False, zorder=20)
                # An external coordinate table avoids covering the measured surface.
                position = ax.get_position()
                for i, (point, marker_color) in enumerate(zip(points, marker_colors)):
                    line = f"{point['id']}: ({point['channel']}, {point['token']}, {number(point['z'])})"
                    fig.text(.98, position.y0 + position.height/2 + (.42-i*.14)/height,
                             line, color=marker_color, ha='right', va='center', fontsize=7)
            records.append({**original, 'shape': [128, 512], 'vertices_processed': int(z.size),
                'surface_cells': 127 * 511, 'surface_rcount': 128, 'surface_ccount': 512,
                'data_stride': [1, 1], 'pooling': False, 'cropping': True,
                'tokens': [0, 128], 'channels': [0, 512], 'draw_surface': True,
                'draw_height_columns': False, 'draw_sidewalls': False,
                'vertical_height_segments': 0, 'sidewall_triangles': 0, 'zero_padding': False,
                'z_limit': limit, 'all_zero_column': maximum == 0,
                'norm': {'name': type(norm).__name__, 'gamma': None if linear else .5,
                         'vmin': 0, 'vmax': limit, 'cmap': 'viridis'},
                'debug_points': points, 'camera': CONFIG['camera']})
        # One horizontal, data-unit colorbar per column, outside the surface grid.
        bottom_ax = axes[rows - 1, col].get_position()
        cax = fig.add_axes([bottom_ax.x0 + .1 * bottom_ax.width, .51 / height,
                            .80 * bottom_ax.width, .07 / height])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=colormaps['viridis']), cax=cax,
                          orientation='horizontal', ticks=[0, limit / 2, limit])
        cb.ax.xaxis.set_major_formatter(ticker.FuncFormatter(number))
        cb.ax.tick_params(labelsize=7, pad=1, length=2)
        cb.outline.set_linewidth(.3)
        colorbars.append(cax)
    fig.canvas.draw()
    if mode == 'end_to_end' and 'unrotated' in methods and len(methods) > 1:
        y = (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
        fig.add_artist(Line2D([.05 / width, .95 / width], [y, y],
                              transform=fig.transFigure, color='.7', linewidth=.5))
    # Exact source coordinates go through the same native camera as the surface.
    for record in records:
        ax = axes[methods.index(record['method']), list(layers).index(record['layer'])]
        for point in record['debug_points']:
            projected = proj3d.proj_transform(point['channel'], point['token'], point['z'], ax.get_proj())
            point['native_projected_xy'] = list(map(float, projected[:2]))
    path = Path(directory) / name if directory else Path(destination) / 'figures' / mode / site / quantity / 'detail' / name
    scales = {'z_limits_by_layer': limits, 'all_zero_by_layer': zero_flags,
              'shared_across_methods': candidates, 'linear_heights': True,
              'norm': 'Normalize' if linear else 'PowerNorm', 'gamma': None if linear else .5,
              'cmap': 'viridis', 'vmin': 0, 'view': 'rotated_only_zoom' if zoom else 'four_method_scale',
              'scale_differs_from_matrix': zoom}
    export(fig, path, records, scales, colorbars)
    print('LOCAL SURFACE', path, flush=True)
    del fig, axes, colorbars
    gc.collect()
    return path


def run(source, destination, selection, parts='all'):
    mode, site, quantity = selection.split('/')
    for name, methods, linear in [('matrix', METHODS, False), ('matrix_linear', METHODS, True),
                                   ('rotated_only_zoom', METHODS[1:], False)]:
        matrix(source, destination, mode, site, quantity, methods, name=name, linear=linear)
    if parts == 'matrix':
        return
    for method in METHODS:
        matrix(source, destination, mode, site, quantity, [method], name=f'row_{method}', scale_methods=METHODS)
        for layer in LAYERS:
            matrix(source, destination, mode, site, quantity, [method], [layer],
                   name=f'panel_{method}_block{layer+1}', scale_methods=METHODS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('destination')
    parser.add_argument('--select', default='end_to_end/q_proj/raw')
    parser.add_argument('--parts', choices=['all', 'matrix'], default='all')
    args = parser.parse_args()
    run(args.source, args.destination, args.select, args.parts)
