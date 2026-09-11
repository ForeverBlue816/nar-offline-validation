# Full-resolution rendering revision

User-directed revision: preserve the complete 2048-token sequence and every
channel (12288 at down_proj; 4096 at q_proj). Do not use the previous 128 x 256
maximum-pooling cache, channel/token strides, smoothing, or a cropped detail
window. The fixed sample remains sample 0 for every method; all eight samples
continue to determine reported metrics.

Plot the actual absolute post-rotation/pre-QDQ values, or the absolute signed
group-centered residual, as a height surface. Close its four exterior boundaries
to z=0, so the display reads as height rising from a base. These geometric
sidewalls are not additional activation observations. Never shift, clip, or
zero measured surface heights. Use shared linear z/color limits across methods
within each layer/site/quantity. Unrotated reference rows remain labelled.

Render every grid vertex and both triangles of every adjacent grid cell through
a depth buffer. A streaming Python/Numba rasterizer avoids allocating tens of
millions of Matplotlib polygon objects. Its camera comes directly from the
Matplotlib axes. Record the number of vertices and triangles, raw shard hash,
array shape, source maximum, and zero base for every panel. Depth-tested vertex
coverage retains even subpixel projected peaks. Pixel resolution and physical
occlusion still limit what can be distinguished on a printed page; neither is
input-data subsampling. SVG/PDF keep axes/text editable and embed the surface
raster at 600 dpi.

Tick labels are real channel/token indices. Channel ticks advance by 2000;
token ticks are 0, 1000, 2000. Exact array domains remain 0..12287/4095 and
0..2047. Typography is Times New Roman. Multicolumn exports are 12 inches wide;
8.5-point tick labels remain at least 5.1 points if reduced to 7.2 inches.

The archived capture and all numerical limitations remain unchanged, including
three failed supplementary frozen-factor checks. This revision changes the
presentation and does not support a new scientific-performance claim.

QA: verify projection, depth ordering, wall closure and complete vertex/cell
counts on small deterministic geometry tests; inspect the complete measured
Block 36 panel before batch rendering; audit panel alignment, PDF text and
collisions; inspect all revised matrix panels. No synthetic test surface may
appear among the scientific figures.

Final height-column rule: every measured vertex also contributes a depth-tested vertical segment from z=0 to its unchanged value. This connects even subpixel-width peaks continuously to the base. The segment represents geometric height, not additional activation observations. The complete grid-cell mesh and exterior closure remain present. Geometry reports record one vertical segment per source element.
