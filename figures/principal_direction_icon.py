#!/usr/bin/env python3
"""Seeded conceptual PCA glyph, not measured PrismQuant activation data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

from figure_style import PALETTE, save_panel
from figure_typography import configure_times_bold

SEED = 20260907
SAMPLES = 72
NAVY = PALETTE['prismquant']
POINTS = PALETTE['prismquant_light']
CONTOUR = PALETTE['hadamard']


def cloud_geometry():
    """Keep every simulated observation and fit its empirical PCA geometry."""
    rng = np.random.default_rng(SEED)
    angle = np.deg2rad(28)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    cloud = (rng.standard_normal((SAMPLES, 2)) * [1.0, .24]) @ rotation.T
    cloud -= cloud.mean(axis=0)
    covariance = np.cov(cloud, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    values, vectors = values[::-1], vectors[:, ::-1]
    if vectors[0, 0] < 0:
        vectors[:, 0] *= -1
    # A consistent secondary arrow points up and left.
    vectors[:, 1] = [-vectors[1, 0], vectors[0, 0]]
    return cloud, covariance, values, vectors


def configure_icon_style():
    font = configure_times_bold(1)
    plt.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.default': 'regular',
                         'mathtext.fallback': None})
    for field in ('rm', 'it', 'bf', 'bfit', 'cal', 'sf', 'tt'):
        plt.rcParams[f'mathtext.{field}'] = 'Times New Roman:bold'
    return font


def draw_icon(ax, center=(0, 0), width=.82, secondary=True, label_size=7.5,
              point_size=2.8, arrow_width=.4):
    cloud, covariance, values, vectors = cloud_geometry()
    std = np.sqrt(values)
    main_start, main_end = -2.45 * std[0] * vectors[:, 0], 2.65 * std[0] * vectors[:, 0]
    # One isotropic scale preserves covariance geometry and orthogonality.
    extent = max(np.abs(cloud[:, 0]).max(), abs(main_start[0]), abs(main_end[0]))
    scale = .5 * width / extent
    origin = np.asarray(center)
    points = origin + scale * cloud
    angle = np.rad2deg(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ax.add_patch(Ellipse(origin, 4 * std[0] * scale, 4 * std[1] * scale,
                         angle=angle, facecolor=PALETTE['zero'], edgecolor=CONTOUR,
                         linewidth=.55, alpha=.65, zorder=1))
    ax.scatter(points[:, 0], points[:, 1], s=point_size, color=POINTS,
               alpha=.56, edgecolors='none', zorder=2, rasterized=False)
    tip = origin + scale * main_end
    ax.add_patch(FancyArrowPatch(origin + scale * main_start, tip,
                                arrowstyle='-|>', mutation_scale=3.2,
                                linewidth=arrow_width, color=NAVY,
                                shrinkA=0, shrinkB=0, zorder=4))
    ax.annotate(r'$v_1$', tip, xytext=(0, 2.2), textcoords='offset points',
                fontsize=label_size, color=NAVY, ha='center', va='bottom', zorder=5)
    if secondary:
        tip2 = origin + scale * (2.15 * std[1] * vectors[:, 1])
        ax.add_patch(FancyArrowPatch(origin, tip2, arrowstyle='-|>', mutation_scale=2.8,
                                    linewidth=.3, color=POINTS, shrinkA=0, shrinkB=0, zorder=3))
        ax.annotate(r'$v_2$', tip2, xytext=(-3, 1.5), textcoords='offset points',
                    fontsize=label_size, color=POINTS, ha='right', va='bottom', zorder=5)
    return {'sample_count': len(cloud), 'covariance_contour_mahalanobis_radius': 2.0,
            'scale': scale, 'center': list(center), 'width': width, 'secondary_direction': secondary}


def main():
    here = Path(__file__).resolve().parent
    font = configure_icon_style()
    cloud, covariance, values, vectors = cloud_geometry()
    fig = plt.figure(figsize=(1.0, 2 / 3), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(-.5, .5), ylim=(-1 / 3, 1 / 3), aspect='equal')
    ax.axis('off')
    geometry = draw_icon(ax, center=(0, -.075), width=.74, secondary=True)
    save_panel(fig, here / 'fig1_principal_directions', dpi=300, axes=[ax])
    np.savetxt(here / 'fig1_principal_directions.csv',
               np.column_stack([np.arange(SAMPLES), cloud]), delimiter=',',
               header='sample_id,simulated_x,simulated_y', comments='', fmt=['%d', '%.17g', '%.17g'])
    metadata = {'kind': 'synthetic conceptual illustration, not measured activations',
                'seed': SEED, 'samples_generated': SAMPLES, 'samples_plotted': SAMPLES,
                'sampling': 'NumPy default_rng; 2D Gaussian standard deviations [1.0, 0.24], rotated 28 degrees; sample centered',
                'empirical_covariance': covariance.tolist(), 'empirical_eigenvalues': values.tolist(),
                'empirical_eigenvectors_columns': vectors.tolist(),
                'principal_energy_fraction': float(values[0] / values.sum()),
                'sample_array_sha256': hashlib.sha256(cloud.tobytes()).hexdigest(),
                'palette': {'principal_direction': NAVY, 'samples': POINTS, 'contour': CONTOUR},
                'font': font, 'geometry': geometry,
                'arrow_linewidth_pt': {'v1': .4, 'v2': .3},
                'output': '1 in by 2/3 in, white background; editable vector SVG and PDF; 300 x 200 PNG'}
    (here / 'fig1_principal_directions_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
