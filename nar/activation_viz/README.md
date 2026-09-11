# Qwen3 activation diagnostics

Read-only analysis of the existing E22 Qwen3-8B-Base experiment. No model training, calibration, benchmark rerun or optimized inference-kernel changes.

```bash
# GPU; smoke checks gate full capture, then independent full-resolution statistics.
python -m nar.activation_viz.capture --workdir "$NAR_WORKDIR" \
  --output "$NAR_WORKDIR/outputs/qwen_activation_viz/qwen3_8b_seed42"
# Statistics can be repeated without loading a model.
python -m nar.activation_viz.metrics "$RUN" --device cuda
# Plotting needs only display_cache.npz and range_ecdf.npz.
python -m nar.activation_viz.plot "$RUN" --select matrices
python -m nar.activation_viz.plot "$RUN"
```

The included SLURM entry point follows the project's existing GPU allocation convention. Python dependencies are the existing experiment environment (torch, transformers, datasets, numpy) plus Matplotlib. Plotting requires installed Times New Roman fonts; install them under `~/.local/share/figure-fonts/times-new-roman/`. This is an explicit typography choice requested for these figures. The static Nature-style checker assumes sans-serif, so its FONT-FAMILY warning is reviewed against this choice; exported glyph audits still apply.

The signed activation shards (several GB) stay on the experiment volume. Git contains SHA-256 inventory, input IDs, full-precision per-sample and pooled summaries, plot caches, paper captions and PDF/SVG/600-dpi PNG exports. Figures never consume synthetic tensors. Small synthetic numerical edge checks, when run, are not figure data.

`qa/panel_alignment.py` is the unmodified alignment utility copied from the installed nature-figure skill; its SHA-256 and rendered reports identify the gate used. The plotting entry point blocks misaligned comparable panels. PDF text-size and collision audits are separate post-export steps, and each warning requires visual review.
