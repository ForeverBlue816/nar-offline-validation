# Qwen3 activation diagnostics

Analysis of the existing E22 Qwen3-8B-Base experiment. No model training, calibration, benchmark rerun or optimized inference-kernel changes.

```bash
# GPU capture; required smoke checks gate complete collection.
python -m nar.activation_viz.capture --workdir "$NAR_WORKDIR" --output "$RAW_RUN"
# Complete statistics can be repeated without loading a model.
python -m nar.activation_viz.metrics "$RAW_RUN" --device cuda
# CPU rendering reads signed raw shards. RUN is a separate figure directory.
python -m nar.activation_viz.full_batch "$RAW_RUN" "$RUN" --workers 8
python -m nar.activation_viz.render_batch "$RUN" --audit-only
python -m nar.activation_viz.full_surface_checks --output geometry_checks.json
python -m nar.activation_viz.summarize "$RUN"
python -m nar.activation_viz.publish "$RUN" "$PUBLICATION_DIR"
```

`slurm_qwen_full_resolution.sh` renders on CPUs only. The validated rendering environment uses Python 3.11, NumPy 2.4.6, Matplotlib 3.10.5, Numba 0.67.0 and llvmlite 0.49.0, plus the existing Torch environment. Install rendering additions in an isolated environment rather than changing experiment dependencies. Plotting requires Times New Roman fonts under `~/.local/share/figure-fonts/times-new-roman/`. This explicit user typography choice overrides the static checker's sans-serif whitelist; exported glyph-size audits still apply.

All 2048 tokens and all 12288 down-projection or 4096 query-projection channels enter each surface. The old pooled display cache is no longer used or distributed. The historical overview/detail URLs both show identical complete-data exports. Figures never consume synthetic tensors; numerical and geometry test arrays are never figure data.

Height surfaces are closed to z=0 at their exterior boundaries. The added sides convey height above the base; they do not change measured surface values or add activation observations. A streaming depth buffer processes every vertex, a zero-to-value vertical height segment at that vertex, and both triangles of every adjacent grid cell. The vertical segments keep subpixel-width peaks connected to the base. Per-panel geometry reports record source hashes and complete element counts. This avoids allocating tens of millions of Python polygon objects while retaining all input data. The output is still a finite-resolution raster with normal perspective occlusion. Text and axes remain editable in PDF/SVG.

Signed activation shards (approximately 30 GB) remain on the experiment volume. Git contains their SHA-256 inventory, input IDs, full-precision per-sample and pooled summaries, exact ECDF values/counts, paper captions and PDF/SVG/600-dpi PNG exports. Exact ECDFs use every distinct range and its full multiplicity, without quantile thinning or path simplification.

The plotting entry point blocks misaligned comparable panels. PDF text-size and collision audits follow export, and warnings require visual review. `qa/panel_alignment.py` is the unchanged utility copied from the nature-figure skill. The published run index contains numerical limitations and final visual-review decisions. `python -m nar.activation_viz.plot "$RUN" --select distributions` updates exact ECDFs only.
