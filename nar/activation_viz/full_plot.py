"""Unpooled, uncropped Qwen activation height figures from signed raw shards."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from .plot import style, CMAP, METHODS, LABELS, LAYERS, number, save
from .full_surface import add_height_surface


def setup_axes(ax, shape, limit):
    rows, cols = shape
    ax.patch.set_alpha(0)
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect((1.28, 1, .72), zoom=.88)
    ax.set_xlim(0, cols-1); ax.set_ylim(0, rows-1); ax.set_zlim(0, limit)
    ax.set_xticks(np.arange(0, cols, 2000))
    ax.set_yticks([0, 1000, 2000])
    ax.set_zticks([0, limit/2, limit])
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(number))
    ax.set_xlabel('Channel', labelpad=19, fontsize=9)
    ax.set_ylabel('Token', labelpad=15, fontsize=9)
    ax.tick_params(axis='both', pad=1, length=2, labelsize=8.5)
    ax.tick_params(axis='z', pad=3, labelsize=8.5)
    ax.tick_params(axis='x', labelrotation=75)
    for label in ax.get_xticklabels():
        label.set_rotation_mode('anchor'); label.set_horizontalalignment('right'); label.set_verticalalignment('top')
    for label in ax.get_zticklabels(): label.set_horizontalalignment('left')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.975, .979, .982, 1))
        axis.line.set_color('#869099'); axis.line.set_linewidth(.5)
        axis._axinfo['grid']['linewidth'] = .3
        axis._axinfo['grid']['color'] = (.81, .84, .86, 1)


def load(root, mode, method, layer, site, quantity):
    path = root/'activations'/mode/method/f's00_l{layer:02d}_{site}.pt'
    y = torch.load(path, map_location='cpu', weights_only=True)
    assert y.shape == (2048, 12288 if site == 'down_proj' else 4096)
    assert y.dtype == torch.float32 and torch.isfinite(y).all()
    if quantity == 'residual':
        groups = y.reshape(y.shape[0], -1, 128)
        y = (groups-groups.mean(-1, keepdim=True)).reshape_as(y)
    return y.abs().numpy(), path


def matrix(root, out, mode, site, quantity, methods, layers=LAYERS,
           name='matrix', scale_methods=None):
    torch.set_num_threads(1)
    nrow, ncol = len(methods), len(layers)
    width = 12 if ncol == 4 else 3.25
    height = 2.65*nrow+.85 if ncol == 4 else 3.5
    fig = plt.figure(figsize=(width, height), dpi=600)
    grid = fig.add_gridspec(nrow, ncol, left=.075 if ncol == 4 else .10,
                           right=.93 if ncol == 4 else .88,
                           bottom=35/(height*72), top=1-44/(height*72),
                           wspace=.18, hspace=.16)
    panels = []; limits = {}; started = time.monotonic()
    candidates = scale_methods or (METHODS if mode == 'paired_local' else METHODS[1:])
    for col, layer in enumerate(layers):
        maximum = max(float(load(root, mode, m, layer, site, quantity)[0].max()) for m in candidates)
        limit = maximum*1.03 or 1.; limits[str(layer)] = limit
        for row, method in enumerate(methods):
            ax = fig.add_subplot(grid[row, col], projection='3d')
            setup_axes(ax, (2048, 12288 if site == 'down_proj' else 4096), limit)
            if row == 0: ax.set_title(f'Block {layer+1}', pad=3, fontsize=11)
            if col == 0:
                fig.text(.016 if ncol == 4 else .03,
                         1-(row+.5)/nrow*.895-.05, LABELS[method], rotation=90,
                         rotation_mode='anchor', va='center', ha='center', fontsize=10)
            panels.append((ax, method, layer, limit))
    title = 'Post-rotation activation magnitude' if quantity == 'raw' else 'Group-centered residual magnitude'
    if 'unrotated' in methods and quantity == 'raw': title = 'Pre-quantization activation magnitude'
    fig.text(.51, 1-4/(height*72), title, ha='center', va='top', fontsize=11)
    fig.text(.51, 1-20/(height*72), 'All tokens and channels; sides extend to zero',
             ha='center', va='top', fontsize=8.5)
    # Freeze layout before projecting measured vertices into each exact camera.
    fig.canvas.draw()
    if ncol == 1:
        renderer = fig.canvas.get_renderer()
        for ax, _, _, _ in panels:
            labels = ax.get_zticklabels()
            if any(label.get_window_extent(renderer).x1 > fig.bbox.x1 for label in labels):
                for label in labels: label.set_horizontalalignment('center')
        fig.canvas.draw()
    records = []
    for ax, method, layer, limit in panels:
        z, source = load(root, mode, method, layer, site, quantity)
        record = add_height_surface(ax, z, limit, CMAP)
        record.update(method=method, layer=layer, source=str(source.relative_to(root)),
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      maximum=float(z.max()), z_limit=limit)
        records.append(record)
        del z
    path = out/mode/site/quantity/'overview'/name
    save(fig, path)
    path.with_suffix('.scales.json').write_text(json.dumps({'z_limits_by_layer': limits,
        'shared_across_methods': list(candidates), 'linear': True}, indent=2)+'\n')
    path.with_suffix('.geometry.json').write_text(json.dumps({'panels': records,
        'sample': 0, 'sidewalls': 'geometric closure to z=0; not additional measurements',
        'seconds': time.monotonic()-started}, indent=2)+'\n')
    print('FULL RESOLUTION', path, time.monotonic()-started, flush=True)


def run(root, destination, selection, single=False, parts='all'):
    root, out = Path(root), Path(destination)/'figures'; style()
    mode, site, quantity = selection.split('/')
    methods = METHODS if mode == 'paired_local' else METHODS[1:]
    if single:
        matrix(root, out, mode, site, quantity, ['nar_kmax'], [35], name='panel_nar_kmax_block36')
        return
    matrix(root, out, mode, site, quantity, methods)
    if mode == 'paired_local':
        matrix(root, out, mode, site, quantity, METHODS[1:], name='rotated_only_zoom', scale_methods=METHODS[1:])
    if parts == 'matrix': return
    for method in methods:
        matrix(root, out, mode, site, quantity, [method], name=f'row_{method}')
        for layer in LAYERS:
            matrix(root, out, mode, site, quantity, [method], [layer], name=f'panel_{method}_block{layer+1}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination')
    p.add_argument('--select', default='end_to_end/down_proj/raw'); p.add_argument('--single', action='store_true')
    a=p.parse_args(); run(a.root, a.destination, a.select, a.single)
