# Figure contracts

## Figure 1 — plus/minus versus common mode

- Core conclusion: A random rotation spreads a concentrated outlier as signed `±c/sqrt(g)` values and inflates the quantization range; NAR aligns it as `+c/sqrt(g)`, making it a groupwise constant that the zero point removes.
- Manuscript role: Page-one mechanism teaser.
- Archetype: Quantitative evidence grid with one dominant hero panel.
- Hero panel: Four paired signed traces from one 128-channel group at the token containing the largest-magnitude raw outlier: identity, random-sign H128, NAR, and NAR after subtracting the group mean. Every pre-subtraction trace includes its group mean and a numeric max–min bracket.
- Support panels: Three signed channel-by-token surfaces with shared z limits; one ECDF over all 64 paired tokens and all channel groups.
- Population and uncertainty: The ECDF is empirical over `64 × (d/128)` paired token-groups; no inferential interval is added.
- Source: Frozen E1c `down_input` dump and frozen E1c transforms for Llama-3.1-8B. Layer selection is deterministic: largest absolute raw activation among the first 64 paired rows across layers. The plotted group is the 128-channel output group containing that selected outlier after NAR mapping.
- Integrity rule: Signed activations only. Group zero point is the arithmetic group mean. Existing result artifacts are read-only.
- Reviewer risk: The numeric 2D hero and full-population ECDF carry the evidence; the surfaces are explanatory rather than the sole support.

## Figure 2 — null-space capture

- Core conclusion: Existing rotations leave most of the affine quantizer null space unused, whereas closed-form NAR fully absorbs the leading activation directions.
- Manuscript role: Section 3 discovery/diagnosis.
- Archetype: Two-panel quantitative comparison grid with shared y axis.
- Hero panel: Both panels are equal-weight; q/k/v input and down-projection input are the two replication sites.
- Encodings: Direction index on x, null-space capture `s_i` on y; mean over layers is a line, and the 10th–90th layer percentile is a light band. Dashed references mark random orthogonal capture `G/d` and full absorption.
- Source: Frozen E16 per-layer DC-alignment CSVs for Llama-3.1-8B and Llama-3.2-3B. Only Hadamard, DuQuant-style, and NAR `k=max` are plotted. Top-32 directions are used if present, otherwise all available top directions.
- Integrity rule: Per-method right-margin means are computed from exactly the plotted layer-direction rows. Existing result artifacts are read-only.
- Reviewer risk: The shared scale, explicit random baseline, percentile bands, and appendix-model replication prevent single-layer or axis-scale effects from driving the claim.
