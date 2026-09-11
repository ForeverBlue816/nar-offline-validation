"""Deterministic geometry checks; no test arrays are used in paper figures."""
import argparse, json
from pathlib import Path
import numpy as np
from .full_surface import rasterize, triangle, project


def run():
    matrix = np.eye(4); matrix[2, 2] = -1
    affine = np.array([15., 20., 15., 20.])
    z = np.full((4, 5), 2., dtype=np.float32); before = z.copy()
    heights, depth = rasterize(z, matrix, affine, 160, 160, False)
    assert np.allclose(heights[np.isfinite(depth)], 2)
    assert np.array_equal(z, before)
    matrix[0, 2] = .4; matrix[1, 2] = .6
    opened, od = rasterize(z, matrix, affine, 160, 160, False)
    closed, cd = rasterize(z, matrix, affine, 160, 160, True)
    assert np.isfinite(cd).sum() > np.isfinite(od).sum()
    wall_values = closed[np.isfinite(cd) & ~np.isfinite(od)]
    assert wall_values.min() >= 0 and wall_values.max() <= 2
    assert np.any(wall_values < 1)
    # All measured vertices are projected, including a subpixel isolated peak.
    peak = np.zeros((3, 3), dtype=np.float32); peak[1, 1] = 9
    ph, pd = rasterize(peak, matrix, np.array([.02, 20., .02, 20.]), 40, 40)
    assert np.nanmax(ph) == 9
    matrix[1, 2] = 100.
    ch, cd = rasterize(peak, matrix, np.array([.02, 20., .02, 20.]), 50, 50)
    assert np.isfinite(cd[20:39, 20]).all()
    assert np.nanmax(ch) == 9
    # Nearer geometry wins regardless of primitive submission order.
    d = np.full((20, 20), np.inf); h = np.full_like(d, np.nan)
    a = np.array([1., 1., -2., 2., 1.]); b = a.copy(); b[0] = 15
    c = a.copy(); c[1] = 15
    triangle(a, b, c, d, h)
    for v in (a, b, c): v[2] = -5; v[3] = 5
    triangle(a, b, c, d, h)
    assert h[3, 3] == 5
    for v in (a, b, c): v[2] = -2; v[3] = 2
    triangle(a, b, c, d, h)
    assert h[3, 3] == 5
    # Projection matches the Matplotlib camera algebra exactly.
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    sample_matrix = np.arange(16, dtype=float).reshape(4, 4)+np.eye(4)
    reference = np.asarray(proj_transform(2., 3., 4., sample_matrix))
    actual = project(2., 3., 4., sample_matrix, np.array([1., 0., 1., 0.]))[:3]
    assert np.allclose(reference, actual, atol=1e-14)
    return {'passed': True, 'constant_height_preserved': True, 'input_unchanged': True,
            'sidewalls_close_to_zero': True, 'near_surface_wins': True,
            'subpixel_peak_vertex_retained': True, 'subpixel_peak_connected_to_base': True, 'matplotlib_projection_matches': True,
            'role': 'synthetic geometry checks only; never scientific figure data'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    result = run(); Path(a.output).write_text(json.dumps(result, indent=2)+'\n'); print(result)
