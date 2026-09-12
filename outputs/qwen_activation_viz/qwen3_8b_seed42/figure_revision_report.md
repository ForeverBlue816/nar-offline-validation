# Qwen3-8B local surface revision

The three priority PDFs now show real local upper surfaces. PDF, PNG and SVG exports are synchronized across all eight combinations of paired_local/end_to_end, q_proj/down_proj and raw/residual. This is a user-requested presentation revision, not a new experiment or a change to the original preregistration.

## Priority figures

- [End-to-end q_proj, raw](figures/end_to_end/q_proj/raw/detail/matrix.pdf)
- [End-to-end down_proj, raw](figures/end_to_end/down_proj/raw/detail/matrix.pdf)
- [Paired-local q_proj, raw](figures/paired_local/q_proj/raw/detail/matrix.pdf)

Each detail directory also contains `matrix_linear`, `rotated_only_zoom`, four method rows and sixteen individual panels. There are 184 revised local figure sets and 340 total sets including unchanged full-domain overviews and exact ECDF figures. Each set includes PDF/PNG/SVG.

## Cause and implementation changes

The preceding full-resolution renderer drew a vertical height segment from every measured vertex to zero and closed the perimeter with walls. These marks filled large projected regions and obscured upper-surface variation. Linear color normalization also compressed the visible low-amplitude range. The batch path copied overview into detail, showing the entire domain instead of the requested local window. The end-to-end method list excluded the available unrotated reference.

`full_surface.py` now separates `draw_surface`, `draw_height_columns` and `draw_sidewalls`; the latter two default to false. Current detail uses the new `detail_plot.py` and native Matplotlib `plot_surface`, without manually projecting a raster onto separate axes. `full_plot.py`, `full_batch.py` and `plot.py` route local rendering independently of overview. The batch process does not recompute metrics or ECDFs. `report.py`, `publish.py`, `render_batch.py` and the new detail checks/delivery tools document and validate local assets, reference sources, immutable experimental records and the publication inventory.

Historical rules remain in the archived section of [full_resolution_contract.md](full_resolution_contract.md), run-manifest display history and historical render provenance. They do not govern current detail. Original overview assets remain byte-identical, retaining their historical geometry and global context.

## Data, geometry and reference

Every surface uses fixed sample 0 (saved WikiText-2 sample index 52), tokens [0,128) and channels [0,512): four complete contiguous g128 groups. Native integer coordinates include all 65,536 vertices and 64,897 adjacent cells. Explicit rcount=128 and ccount=512 prevent default sampling. There is no pooling, stride, padding, smoothing, minimum subtraction or manufactured texture. Abrupt measured group steps remain visible where present.

Raw heights are `abs(Y[0:128,0:512])`. Residuals subtract the signed per-token contiguous-g128 mean on complete signed Y, then crop and take absolute values. Residuals are not QDQ reconstruction errors; these means are not actual quantizer offsets. Source arrays are never modified.

All methods share orthographic projection, elevation 28 degrees, azimuth -55 degrees and box aspect (2,1.2,1). Axes span channels 0–511 and tokens 0–127. Real-data previews covered paired_local/down_proj Block 24 raw/residual and end_to_end/q_proj Block 13 with reference. Debug exports label four corners and real peaks; source values and native projected coordinates are recorded in geometry files. No raster is positioned independently of its axes.

Main matrices have four rows and four columns (Blocks 1, 13, 24, 36). The reference is the existing paired_local/unrotated cache: BF16 checkpoint values loaded and norm-fused in FP32, with no rotation or quantization. Input IDs, sample, site, layer and norm-fusion semantics were checked. In end-to-end matrices this unquantized reference is separated from the three existing QDQ forwards; identical intermediate inputs across those four rows are not claimed. Already rotated end-to-end activations are not rotated again. See the [reference audit](qa/detail_reference_audit.json).

## Color and paper layout

Heights remain linear. Main figures use viridis and PowerNorm(gamma=0.5), changing colors only. Each column shares z/color maximum = 1.03 times the largest value among its four methods, with its own true-unit colorbar. All-zero handling is explicit. Native surface face colors represent the mean of their vertices; measured vertex heights remain unchanged.

The fixed `matrix_linear` control uses identical arrays, camera and z limits with Normalize. `rotated_only_zoom` uses the three rotated methods and their shared limits, explicitly labeled as a different scale from the four-row matrix. No method-specific normalization or gamma tuning is used.

Matrices retain 7.02 pt minimum text at 183 mm insertion width; single panels retain 7 pt at 3.5 inches. Channel ticks 0, 256, 511 and token ticks 0, 64, 127 reduce crowding without discarding data. Light floor guides and short axis-edge marks locate boundaries at 127.5, 255.5 and 383.5. Shared front/depth axis labels and per-column amplitude colorbars avoid crowded repeated labels. White/light-gray panes replace colored bases. PDF/SVG axes and text are vector; surfaces are rasterized at 600 dpi. PNGs come directly from final PDFs at 600 dpi.

## Validation and limitations

- All 448 original activation shard hashes and 110 unique frozen-factor hashes match. All 1,253 protected artifact/hash entries match, including numerical CSVs, exact ECDFs, original overview assets, validation records and scientific run-manifest content.
- All 608 repeated local panel records match real source windows, correct residual order, corners/peaks, complete meshes and shared scales. Eight linear controls match the main matrices' data, camera and limits.
- All 184 final local PDFs were rerendered to PNG and checked for paper-size typography. All eight main matrices (128 panels), eight linear controls and eight zoom matrices were visually inspected. Repeated row/panel layouts were spot-checked.
- The publication audit index covers 340 PDFs, with zero blocking font/collision failures. Historical overview/ECDF audit records remain unchanged. Native 3D pane/image bounding-box warnings were visually reviewed and retained.
- Initial zoom subtitle/header collisions were repaired by adding 0.25 inches above the plot grid. All eight repairs preserve arrays, camera, projected coordinates and limits. Initial layout failures are archived separately from the passing delivery review.

The original three supplementary numerical failures remain at their original thresholds. Overall experimental validation remains false; required checks remain true. No activation, rotation factor, quantizer implementation, experimental metric, measured conclusion or downstream-accuracy claim was changed. This fixed local window need not contain global maxima or all model outliers. Similar raw magnitudes remain similar in the plots.

Evidence: [integrity](qa/detail_integrity.json), [final PDF rerenders](qa/detail_pdf_render_index.json), [delivery review](qa/detail_delivery_review.json), [zoom repair](qa/zoom_spacing_repair.json), [render settings](detail_render_config.json), [publication inventory](publication_manifest.json).

## Reproducibility

Renderer implementation: `f7d1795`; paper-size typography: `503f442`; final zoom spacing: `4ba901a`. Slurm job 150482 rendered all local assets with `503f442`; job 150516 regenerated eight zoom sets with `4ba901a`. Full commit IDs, source hashes, runtime/font versions and output paths are recorded in [render_provenance.json](render_provenance.json), [detail_render_config.json](detail_render_config.json) and adjacent geometry files. The final Git artifact commit contains this report and publication inventory; it is distinct from the rendering commits.

The [README](README.md) gives reproduction commands. Raw tensors stay at the external activation root; this publication includes hashes rather than copied tensors. Original preflight previews and superseded layout checks remain under `qa/` with their historical scope identified.

## Priority height-axis follow-up

At the user's request, only the three priority raw `matrix` sets linked above
receive this further layout change. Each panel now labels the linear height
axis at zero, half the shared maximum and the shared maximum, in measured
activation units. The figure identifies the pale floor as z=0 and uses lighter
floor borders. Real positive surfaces still remain above zero; no base is
filled beneath them. Wider numeric labels receive additional tick padding.

All 48 source surfaces, their native projection, shared z/color limits and
complete meshes are unchanged. The footer has 0.25 inches of additional space;
plot-area dimensions and 7.02 pt text at 183 mm remain the same. All other 337
figure sets, including the remaining matrices, linear-color controls, zooms,
rows, individual panels, overviews and ECDFs, remain byte-identical.

The scoped [revision audit](qa/height_axis_revision/audit.json) records the
three new PDF audits, source/geometry equivalence, final-size visual review
and hashes of the protected figures. [PDF render records](qa/detail_pdf_render_index.json)
point to the new pages. The earlier revision report and reviews remain as
history; this scoped record supersedes them only for these three matrices.

Renderer commit: `e2a31a8`; Slurm render job: `150582`. Reproduce with
`python -m nar.activation_viz.height_axis_revision "$RAW_RUN" "$RUN"`.

## Original blue-orange palette restored

The user prefers the original blue-orange appearance. Only the same three
priority raw main matrices are recolored using the existing `plot.CMAP`
(`blue_warm_orange`) with its original six color stops. The common
PowerNorm(gamma=0.5) is retained to make low-amplitude color structure easier
to read. This supersedes the requested Viridis palette for those three main
matrices only. Height ticks, the explicit z=0 floor cue, light borders, exact
panel positions, camera, linear heights and shared limits are preserved.

No data, activation transform or experimental conclusion changes. All 48
panel arrays and their geometry are compared with the preceding height-axis
version. All other 337 figure sets retain their existing files and palette.
In particular, the retained linear-color and zoom companions still use
Viridis; they are not a matched-colormap control for the newly restored
blue-orange main matrices.

The current scoped audit and final visual review are in
[blue_orange_revision/audit.json](qa/blue_orange_revision/audit.json), with
updated final PDF renders in the existing PDF render index. Previous visual
reviews remain historical for these three pages. Renderer commit `3849efd`,
Slurm job `150750`; reproduce with
`python -m nar.activation_viz.height_axis_revision "$RAW_RUN" "$RUN"`.
