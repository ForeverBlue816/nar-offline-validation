# Archived Figure 4 presentation

Frozen at 6747960 before the E28 deployment revision. Original CSVs and component exports remain unchanged in figures/.

## Figure 4

`fig4a` (3B metadata budget), `fig4b` (recovery and kernel cost), and
`fig4a_8b` (appendix) are separate 2.7-by-4.2-inch SVG/PDF/300-dpi PNG panels.
`fig4_preview.png` and `fig4.pdf/svg` combine the two main panels. The figures
contain no titles, footers, protocol notes, or missing-data notices. The
requested short deployment interpretation appears inside panel b.

The budget panels retain every requested existing E20 configuration, with
mean PPL recomputed from three seeds on 64 chunks and effective bits
`4 + 16*(m+1)/g`. The unaccounted fp32 coefficient controls are excluded.
Each panel's CSV lists the exact summary-file row and physical CSV line for
every point and the bf16 reference. Bracket endpoints are those same plotted
points; their source lines and PPL differences are in `fig4_metadata.json`.

The revised presentation keeps both columns at 2.7 by 4.2 inches. Panel a
spans the two stacked panels at right: b1 shows recovery, b2 shows kernel
cost. Both right-hand axes have the same plot width as a, share their x
limits, and use a 62:38 ratio of plot heights. Their combined top and bottom
edges align with a. The two legends occupy a dedicated bottom row with
matching rectangular frames and 6.5-pt type. Only the three model entries
appear in the right-hand legend; b2 identifies the timing models directly.

The separate bare exports `fig4b1` and `fig4b2` have widths of 2.7 inches
and heights of 2.604 and 1.596 inches (62:38, summing to a's 4.2 inches).
They preserve the marks and font sizes while giving each panel its own
margins. `fig4b` retains the assembled right column with its shared legend.
`fig4b1.csv` and `fig4b2.csv` partition the unchanged `fig4b.csv`, preserving
source rows for all 15 recovery points, four kernel points and two references.

Budget tick labels are horizontal; the 4.1875 tick remains without text.
The scale-resolution bracket spans the Hadamard (256,1)/(128,1) pair with
its label above and no leader. The null-space bracket lies at x=4.25; its
label sits to the right, clear of the (256,3) point label. A small stroke
gap at the intervening measured triangle avoids crossing the marker.
The 3B lower bound stays at 7.60, and the measured bf16 line at 7.61675.
The 8B appendix uses the same presentation.

The green deployed band extends through both right-hand axes. Recovery
labels (0.39, 0.36, 0.59) sit to the left of k=8, staggered for clearance.
The two-line interpretation sits at the upper left of b1. Only b2 carries
x tick labels and the shared x-axis label. Its two solid red diamond lines
connect measured k=8 and k=32 costs, with percentages beside the markers
and model names at the ends. Dashed Hadamard references have right-aligned
labels. All source CSV numbers remain unchanged by this layout revision.

The recovery panel evaluates `(mean PPL_Hadamard - mean PPL_k) /
(mean PPL_Hadamard - PPL_bf16)`, directly from PPLs. It does not reuse legacy
recovery columns with different averaging or rotation-only corrections.
Llama uses the E11 activation-only protocol (64 chunks; seeds 20260902,
20260903, 20260904). Qwen uses the existing E18 v2 activation-only results
(146 chunks; seed 20260902). All five requested categories are measured for
all three models; there are no missing k values and no interpolated points.
The per-site maxima (qkv/down) are 24/64 for 3B, 32/112 for Llama 8B,
and 32/96 for Qwen 8B. At each nominal k, the actual site value is capped
by that site's maximum. The CSV records both values.

The previously absent Llama k=64 rows were measured on September 5, 2026,
using the original E11 code at `f424b82`, original test tokens and rotation
seeds, bf16 weights/KV, and group-128 activation quantization at both sites.
The frozen E11 eigenvectors were recovered by inverting its b64 Householder
factors, checked against the stored b128 k32 reflectors, then used for a
fresh original-protocol 128-sequence permutation calibration. There is no
new eigensolver fit. Raw losses, factor/basis audits, summary PPLs, and device
and token-hash provenance are in `results/<model>/e11_k64_*` and
`E11_K64_DONE.json`. Original E11 tables are preserved.

Kernel costs come from E17 v3's `*_share_of_layer` columns:
`100 * kernel_ms / (decoder_layer_ms + kernel_ms)`. The k=8 timing row
supplies each model's single horizontal Hadamard reference. PrismQuant has
only the measured k=8 and k=32 timing points; no costs are estimated at other
k values. `fig4b.csv` carries separate source files and exact CSV line numbers
for every recovery value, each baseline, and each kernel measurement.

To reproduce the exports and independently verify raw-loss and CSV linkage:

```bash
python figures/make_fig4.py
python figures/verify_fig4.py
python figures/audit_exports.py --figure 4
```

To repeat a missing-k experiment, export the `nar/` directory from commit
`f424b82` into a separate code directory, then run on an allocated GPU:

```bash
python figures/measure_fig4_k64.py --model llama32_3b \
  --repo "$PWD" --assets "$NAR_WORKDIR" --scratch "$FIG4_SCRATCH" \
  --code-root "$FIG4_FROZEN_CODE"
```

Use `llama31_8b` for the other model. Existing completed measurements are
reused. The required frozen factors and model cache reside in the asset root.

