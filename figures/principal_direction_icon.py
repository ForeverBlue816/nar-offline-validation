#!/usr/bin/env python3
"""Seeded conceptual PCA glyph, not measured PrismQuant activation data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

from figure_style import save_panel
from figure_typography import configure_times_bold

SEED = 20260907
SAMPLES = 216
NAVY = '#4D6694'
POINTS = '#9FBC89'
POINT_EDGE = '#567650'
CONTOUR = '#86A875'
TEAL = '#539B8B'
OUTER_FILL = '#ECF3E7'
INNER_FILL = '#D1E6DC'


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
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                         'mathtext.fontset': 'custom', 'mathtext.default': 'regular',
                         'mathtext.fallback': None})
    for field in ('rm', 'it', 'bf', 'bfit', 'cal', 'sf', 'tt'):
        plt.rcParams[f'mathtext.{field}'] = 'Times New Roman:bold'
    return font


def draw_icon(ax, center=(0, 0), width=.82, secondary=True, label_size=7.5,
              point_size=4.2, arrow_width=1.15):
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
                         angle=angle, facecolor=OUTER_FILL, edgecolor=CONTOUR,
                         linewidth=.95, alpha=.95, zorder=1))
    ax.add_patch(Ellipse(origin, 2 * std[0] * scale, 2 * std[1] * scale,
                         angle=angle, facecolor=INNER_FILL, edgecolor=TEAL,
                         linewidth=.75, linestyle=(0, (2.5, 2)), alpha=.85, zorder=1.5))
    # Keep all simulated points, including the tail observations, inside the panel.
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    assert np.all(points.min(axis=0) > limits[:, 0] + .02)
    assert np.all(points.max(axis=0) < limits[:, 1] - .02)
    ax.scatter(points[:, 0], points[:, 1], s=point_size, color=POINTS,
               alpha=.82, edgecolors=POINT_EDGE, linewidths=.22, zorder=2, rasterized=False)
    tip = origin + scale * main_end
    ax.add_patch(FancyArrowPatch(origin + scale * main_start, tip,
                                arrowstyle='-|>', mutation_scale=6.0,
                                linewidth=arrow_width, color=NAVY,
                                shrinkA=0, shrinkB=0, zorder=4))
    ax.annotate(r'$v_1$', tip, xytext=(6.0, 0), textcoords='offset points',
                fontsize=label_size, color=NAVY, ha='left', va='center', zorder=5)
    if secondary:
        tip2 = origin + scale * (2.15 * std[1] * vectors[:, 1])
        ax.add_patch(FancyArrowPatch(origin, tip2, arrowstyle='-|>', mutation_scale=5.5,
                                    linewidth=.95, color=TEAL, shrinkA=0, shrinkB=0, zorder=3))
        ax.annotate(r'$v_2$', tip2, xytext=(-4.5, 4.0), textcoords='offset points',
                    fontsize=label_size, color=TEAL, ha='right', va='bottom', zorder=5)
    return {'sample_count': len(cloud), 'covariance_contour_mahalanobis_radii': [1.0, 2.0],
            'scale': scale, 'center': list(center), 'width': width, 'secondary_direction': secondary}


def main():
    here = Path(__file__).resolve().parent
    font = configure_icon_style()
    cloud, covariance, values, vectors = cloud_geometry()
    fig = plt.figure(figsize=(2.0, 1.0), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(-1.0, 1.0), ylim=(-.5, .5), aspect='equal')
    ax.axis('off')
    geometry = draw_icon(ax, center=(0, 0), width=1.34, secondary=True, label_size=9.5)
    save_panel(fig, here / 'fig1_principal_directions', dpi=600, axes=[ax])
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
                'palette': {'principal_direction': NAVY, 'secondary_direction': TEAL,
                            'samples': POINTS, 'sample_edges': POINT_EDGE, 'outer_contour': CONTOUR,
                            'outer_fill': OUTER_FILL, 'inner_fill': INNER_FILL},
                'color_semantics': 'One simulated population; colors distinguish sample marks, geometric contours, and PCA arrows, not classes.',
                'font': font, 'geometry': geometry,
                'arrow_linewidth_pt': {'v1': 1.15, 'v2': .95},
                'revision': {'previous_samples': 72, 'current_samples': SAMPLES,
                             'sampling_law_unchanged': True, 'previous_canvas_aspect': 1.5, 'current_canvas_aspect': 2.0},
                'contour_interpretation': 'Mahalanobis radii 1 and 2 of empirical covariance; not confidence intervals.',
                'output': '2 in by 1 in, white background; editable vector SVG and PDF; 1200 x 600 PNG at 600 dpi'}
    (here / 'fig1_principal_directions_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
