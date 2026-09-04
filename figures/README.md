Naming: the method is PrismQuant in the paper. Code, CSV method columns and artifact filenames retain the development name 'nar' (nar_k8 = PrismQuant k=8, nar_kmax = PrismQuant k=max, etc.). The development abbreviation was retired because 'NAR' is the established abbreviation for non-autoregressive in NLP.

# Publication figures

Figures 1–3 use the Python/matplotlib backend at the requested final width of 5.5 in. The required serif stack is `Times New Roman`, `Nimbus Roman`, `Liberation Serif`, `DejaVu Serif`; Matplotlib's actual resolved family on this machine is **DejaVu Serif** for every figure. SVG text remains editable (`svg.fonttype='none'`), PDF fonts use Type 42, ordinary text is black, and method labels retain their series color. No visible figure label uses the retired development name.

Two palettes are retained for visual choice:

- Variant A (Okabe–Ito): identity/raw `#7F7F7F`, Hadamard `#E69F00`, DuQuant-style `#009E73`, PrismQuant `#0072B2`.
- Variant B (CAM02-UCS muted): identity/raw `#858484`, Hadamard `#FFE39C`, DuQuant-style `#8DBEA6`, PrismQuant `#001638`.

Variant B is derived programmatically rather than selected from a palette. It is the only variant that passes the hard all-pairs CAM02-UCS ΔE ≥ 15 gate under normal vision, deuteranopia, protanopia, tritanopia and grayscale. Variant A remains available exactly as requested. See `palette_report.md`, `palette_check.png`, and the full-figure simulations under `accessibility/`.

## Figure 1 — persistent outliers become removable common mode

- Main evidence: frozen Llama-3.2-3B E1c `down_input`, layer 1, 1,024 stride-32 rows after excluding sequence position 0 from every sequence, all 8,192 channels. Channel 1419 has the largest non-BOS per-channel median absolute activation. The paired hero is sequence 10, token position 1760.
- Trace ranges: raw 0.521, random-sign block-H128 0.166, PrismQuant k=max 0.100, and PrismQuant after group-zero-point removal 0.100. Mean subtraction recenters the plateau without changing max–min range; the honest residual is non-zero.
- Rendering: the raw panel has its own signed z limit; panels b–d share the random-rotation ±0.202 limit. Every 3D surface uses its own full-absmax-centered `coolwarm` normalization with no clipping. Dense 3D line/tick layers are rasterized inside the SVG/PDF container; panel titles, letters, trace annotations and footer remain editable text.
- Outputs: `fig1_variant{A,B}.{svg,pdf,png}`, canonical Variant-A `fig1_pm_vs_plus.{svg,pdf,png}`, `fig1_ranges.csv`, `fig1_landscape_channels.csv`, caption, metadata and per-variant QA artifacts.
- Font/QA: DejaVu Serif; render-time panel alignment PASS for both variants; PDF-text minimum 7.0 pt with zero violations. Collision audit PASS for both variants with zero failures and zero warnings.

## Figure 2 — per-layer causal chain

- Main figure: Llama-3.2-3B `down_input`; appendix versions cover 3B q/k/v and both 8B sites from measured rows, with no interpolation or cross-model copying.
- Main result: mean whole-activation null-space fractions are 0.00735 / 0.09987 / 0.33279 for Hadamard / DuQuant-style / PrismQuant. PrismQuant reduces mean group range by 25.3% and dynamic asymmetric INT4 NMSE by 40.4% across layers. The 8B down-input appendix gives 25.4% and 40.2%.
- Outputs: `fig2_variant{A,B}.{svg,pdf,png}`, canonical Variant-A `fig2_null_space_capture.{svg,pdf,png}`, three measured appendix renderings, `fig2_capture.csv`, captions, metadata and QA artifacts.
- Font/QA: DejaVu Serif; both main variants pass render-time panel alignment and collision audit, with PDF minimum text 7.0 pt. Appendix audit results are recorded in their JSON files.

## Figure 3 — geometry and square-root range law

- Panel a recomputes the eigenspace from 262,016 non-BOS rows with seed 20260902, oversampling 16, one power iteration and three covariance passes. The preimage of a null-space direction has absolute cosine 1.000 with v₁ for PrismQuant versus 0.005 for Hadamard; their in-plane lengths are 1.000 and 0.012.
- Panel b shows measured rank-256 cumulative energy for down-input layers 1/13/27. The 64-slot cumulative fractions are 0.219/0.249/0.716; their rank-256 fractions are 0.340/0.387/0.801.
- Panel c pools 2,912 labeled rows from E1c activations, E7 V cache and E20 multi-slot diagnostics. Every y value is the group range divided by the paired Hadamard range at the same model/site/layer/group size. The pooled fit is `range_ratio = 0.060 + 0.867 sqrt(1-f)`, R² = 0.861.
- Outputs: `fig3_variant{A,B}.{svg,pdf,png}`, canonical Variant-A `fig3_geometry.{svg,pdf,png}`, `fig3_token_projections.csv`, `fig3_eigenspace_r256.csv`, `fig3_range_law.csv`, caption, metadata and QA artifacts. The activation-derived binary eigenvector cache remains local and is reproducible with `prepare_fig3.py`; it is intentionally excluded from the public repository.
- Font/QA: DejaVu Serif; both variants pass render-time panel alignment, contain no blocking collision findings, and have PDF minimum text 7.0 pt. Remaining WARN entries are raster-scatter/text-edge candidates inspected manually at final physical size.

## QA interpretation

The requested full QA chain is run for each figure: static `validate_figure.py`, render-time `require_matplotlib_panel_alignment()`, `audit_pdf_text.py`, and `audit_figure_collisions.py`, followed by final-size visual inspection. The static validator's built-in font-family rule accepts only sans-serif families, so it reports one intentional FONT-FAMILY failure against the user-mandated serif contract; actual resolution is independently asserted as DejaVu Serif. Its 139.7 mm width and 300 dpi/TIFF notices are likewise expected because the explicit deliverable is 5.5 in SVG/PDF/PNG. Figure 1 also triggers a generic aggregate-without-error-bars warning from the words seed/median/mean; the panel is a deterministic paired representative view rather than an uncertainty summary, as recorded in its contract. Rendered alignment, text-size and collision reports are authoritative for layout.

## Reproduce

```bash
python figures/make_fig1.py --workdir /projects/nar/nar-validation
python figures/make_fig2.py --figure-stats /projects/nar/nar-validation/figure_stats_v2
sbatch slurm_fig3_prepare.sh  # only needed to recompute the frozen BOS-excluded eigenspace
python figures/make_fig3.py
python figures/check_palette.py
```

The scripts write figure-specific data outside `results/`; all existing `results/` files and frozen activation dumps are read-only.
