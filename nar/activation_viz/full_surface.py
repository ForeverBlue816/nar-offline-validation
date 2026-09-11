"""Full-grid height rasterization, with opt-in columns and sidewalls.

Every measured vertex and every adjacent grid cell is processed. There is no
stride, pooling, smoothing, or data selection. Matplotlib supplies the camera,
axes, typography and exports; Numba accelerates the Python rasterization loops.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def project(x, y, z, matrix, affine):
    w = matrix[3, 0]*x + matrix[3, 1]*y + matrix[3, 2]*z + matrix[3, 3]
    px = (matrix[0, 0]*x + matrix[0, 1]*y + matrix[0, 2]*z + matrix[0, 3])/w
    py = (matrix[1, 0]*x + matrix[1, 1]*y + matrix[1, 2]*z + matrix[1, 3])/w
    depth = (matrix[2, 0]*x + matrix[2, 1]*y + matrix[2, 2]*z + matrix[2, 3])/w
    return np.array([px*affine[0]+affine[1], py*affine[2]+affine[3], depth, z/w, 1/w])


@njit(cache=True)
def put_vertex(v, depth, height):
    x = int(np.floor(v[0])); y = int(np.floor(v[1]))
    if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0] and v[2] < depth[y, x]:
        depth[y, x] = v[2]; height[y, x] = v[3]/v[4]


@njit(cache=True)
def height_column(base, top, depth, heights):
    """Depth-tested vertical height segment, including subpixel-width peaks."""
    steps = max(1, int(np.ceil(max(abs(top[0]-base[0]), abs(top[1]-base[1])))))
    for i in range(steps+1):
        t = i/steps
        px = base[0]+t*(top[0]-base[0]); py = base[1]+t*(top[1]-base[1])
        x = int(np.floor(px)); y = int(np.floor(py))
        if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
            d = base[2]+t*(top[2]-base[2])
            if d < depth[y, x]:
                depth[y, x] = d
                heights[y, x] = (base[3]+t*(top[3]-base[3]))/(base[4]+t*(top[4]-base[4]))


@njit(cache=True)
def triangle(a, b, c, depth, height):
    denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(denominator) < 1e-14:
        return
    xmin = max(0, int(np.ceil(min(a[0], b[0], c[0])-.5)))
    xmax = min(depth.shape[1]-1, int(np.floor(max(a[0], b[0], c[0])-.5)))
    ymin = max(0, int(np.ceil(min(a[1], b[1], c[1])-.5)))
    ymax = min(depth.shape[0]-1, int(np.floor(max(a[1], b[1], c[1])-.5)))
    for y in range(ymin, ymax+1):
        for x in range(xmin, xmax+1):
            wa = ((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/denominator
            wb = ((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/denominator
            wc = 1-wa-wb
            if wa >= -1e-9 and wb >= -1e-9 and wc >= -1e-9:
                d = wa*a[2]+wb*b[2]+wc*c[2]
                if d < depth[y, x]:
                    depth[y, x] = d
                    height[y, x] = (wa*a[3]+wb*b[3]+wc*c[3])/(wa*a[4]+wb*b[4]+wc*c[4])


@njit(cache=True)
def wall(x0, y0, z0, x1, y1, z1, matrix, affine, depth, height):
    a = project(x0, y0, 0., matrix, affine)
    b = project(x1, y1, 0., matrix, affine)
    c = project(x1, y1, z1, matrix, affine)
    d = project(x0, y0, z0, matrix, affine)
    triangle(a, b, c, depth, height); triangle(a, c, d, depth, height)


@njit(cache=True)
def rasterize(z, matrix, affine, width, height_px, draw_surface=True,
              draw_height_columns=False, draw_sidewalls=False):
    depth = np.full((height_px, width), np.inf)
    heights = np.full((height_px, width), np.nan)
    rows, cols = z.shape
    previous = np.empty((cols, 5)); current = np.empty((cols, 5))
    for y in range(rows):
        for x in range(cols):
            current[x] = project(x, y, z[y, x], matrix, affine)
            # Preserve measured vertices even for subpixel projected triangles.
            if draw_surface:
                put_vertex(current[x], depth, heights)
            if draw_height_columns:
                height_column(project(x, y, 0., matrix, affine), current[x], depth, heights)
        if y and draw_surface:
            for x in range(cols-1):
                triangle(previous[x], previous[x+1], current[x+1], depth, heights)
                triangle(previous[x], current[x+1], current[x], depth, heights)
        previous, current = current, previous
    if draw_sidewalls:
        for x in range(cols-1):
            wall(x, 0, z[0, x], x+1, 0, z[0, x+1], matrix, affine, depth, heights)
            wall(x, rows-1, z[-1, x], x+1, rows-1, z[-1, x+1], matrix, affine, depth, heights)
        for y in range(rows-1):
            wall(0, y, z[y, 0], 0, y+1, z[y+1, 0], matrix, affine, depth, heights)
            wall(cols-1, y, z[y, -1], cols-1, y+1, z[y+1, -1], matrix, affine, depth, heights)
    return heights, depth


def add_height_surface(ax, z, limit, cmap, *, norm=None, draw_surface=True,
                       draw_height_columns=False, draw_sidewalls=False):
    """Attach a 600-dpi full-grid raster in the exact Matplotlib camera frame."""
    from matplotlib.image import BboxImage
    bbox = ax.bbox
    width, height = int(np.ceil(bbox.width)), int(np.ceil(bbox.height))
    transform = ax.transData.get_affine().get_matrix()
    affine = np.array([transform[0, 0], transform[0, 2]-bbox.x0,
                       transform[1, 1], transform[1, 2]-bbox.y0])
    from matplotlib.colors import Normalize
    norm = norm if norm is not None else Normalize(0, limit)
    values, depth = rasterize(z, ax.get_proj(), affine, width, height,
                              draw_surface, draw_height_columns, draw_sidewalls)
    rgba = cmap(norm(np.nan_to_num(values)), bytes=True)
    rgba[..., 3] = np.where(np.isfinite(depth), 255, 0)
    image = BboxImage(bbox, interpolation='none', origin='lower', zorder=2)
    image.set_data(rgba); ax.add_artist(image)
    return {'shape': list(z.shape), 'vertices_processed': int(z.size),
            'surface_triangles': int(2*(z.shape[0]-1)*(z.shape[1]-1)) if draw_surface else 0,
            'sidewall_triangles': int(4*(z.shape[0]+z.shape[1]-2)) if draw_sidewalls else 0,
            'draw_surface': draw_surface, 'draw_height_columns': draw_height_columns,
            'draw_sidewalls': draw_sidewalls,
            'data_stride': [1, 1], 'pooling': False, 'cropping': False,
            'base_z': 0, 'vertical_height_segments': int(z.size) if draw_height_columns else 0, 'pixel_size': [width, height], 'rasterizer': 'depth-tested full grid'}
