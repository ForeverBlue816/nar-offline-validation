# Publication figures

All figures use the Python backend at the requested ICLR text width of 5.5 in. SVG text remains editable (`svg.fonttype = none`), and the minimum rendered PDF text size is 7 pt. Method colors are fixed across the bundle: identity `#7F7F7F`, Hadamard `#E69F00`, DuQuant-style `#009E73`, and NAR `#0072B2`.

## Figure 1 — plus/minus versus common mode

- Contract: A random-sign H128 spreads the selected spike into signed values with an almost-zero group mean, whereas NAR maps it to a common-mode plateau removed by the affine zero point. The population panel reports all `64 × 64 = 4,096` paired token-groups.
- Archetype: Quantitative evidence grid with a four-trace hero, three supporting signed surfaces, and one ECDF.
- Hero: Panels a–d. For the selected token, identity has range 498.01 and group mean −3.89; H128 has range 88.05 and mean approximately zero; NAR's selected DC group has mean −44.14 and residual range 0.000237 before and after mean subtraction.
- Population result: Mean range is 0.101772 for random-sign H128 and 0.074781 for NAR after group-mean subtraction, a paired reduction of 26.5% on the plotted slice.
- Model/layer: Llama-3.2-3B, `down_input`, layer 1, selected from the frozen calibration statistics by the largest absolute per-channel activation (498 at channel 1419). The token-level maximum in the first 64 rows occurs at token 0.
- Data artifacts consumed: `activations/llama32_3b/wide_cal_a/DONE.json`, `wide_cal_a/dumps/down_input/layer_01.bf16`, `e11_calibration/channel_stats.pt`, and `activation_factors/down_layer_01.pt` under the project work directory.
- Source boundary: The requested Llama-3.1-8B token dump does not exist in retained project storage. Only its streamed factors/statistics remain. The figure therefore uses the sole complete frozen E1c token dump (3B) and labels that substitution in the figure; no 3B value is presented as 8B.
- Outputs: `fig1_pm_vs_plus.{svg,pdf,png}`, `fig1_ranges.csv`, `fig1_metadata.json`, source script, alignment JSON/overlay, PDF-text JSON, and collision JSON/overlay.
- QA: Static source validation is ready; panel alignment passes with 8 comparisons and no exemptions; PDF text audit finds a 7.0 pt minimum and zero violations; collision audit passes with zero failures/warnings. Every panel and the full 5.5 in rendering were visually reviewed. Surface colors use a documented ±44.0 robust clip while all three geometric z limits remain the identical ±498 required by the contract.

## Figure 2 — null-space capture

- Contract: Plot `s_i = ||P_DC R v_i||² / ||v_i||²` for the top eight retained second-moment directions at q/k/v and down-projection inputs. Lines are layer means; bands are layer-wise 10th–90th percentiles; the two dashed references are random orthogonal capture `G/d = 0.0078` and full absorption.
- Archetype: Two equal-weight quantitative panels sharing y. There is no single hero panel because the two activation sites are replications of the same diagnostic.
- Main result (Llama-3.1-8B, 32 layers): Mean capture at q/k/v is 0.00539 / 0.52487 / 1.00000 for Hadamard / DuQuant-style / NAR. At `down_input` it is 0.00691 / 0.05536 / 1.00000. Thus DuQuant-style's roughly half-space capture is site-specific; at down input it leaves most of the null space empty, while NAR remains fully aligned.
- Appendix result (Llama-3.2-3B, 28 layers): q/k/v means are 0.00241 / 0.40176 / 1.00000; down-input means are 0.01218 / 0.05749 / 1.00000.
- Data artifacts consumed: `results/<model>/e16_dc_alignment_per_layer.csv`; all frozen `activations/<model>/activation_factors/{qkv,down}_layer_*.pt`; and `activations/<model>/e11_calibration/channel_stats.pt` for `llama31_8b` and `llama32_3b`.
- Derivation check: The missing E16 down-site rows are computed offline from the frozen factors and the exact E11 Hadamard/DuQuant-style constructions. Recomputed q/k/v values agree with the released E16 CSV to maximum absolute error `1.49e-6` (8B) and `1.31e-6` (3B) before the derived down rows are accepted.
- Outputs: `fig2_null_space_capture.{svg,pdf,png}`, appendix `fig2_null_space_capture_3b.{svg,pdf,png}`, shared `fig2_capture.csv`, metadata JSON, source script, and per-render alignment/text/collision QA files.
- QA: Static validation is ready; both alignment reports pass with no exemptions; both PDFs have a 7.0 pt minimum and zero text-size violations. Collision audits have zero failures. The remaining non-blocking fill-edge warnings (three in 8B, two in 3B) are the uncertainty-band edge near explanatory/direct labels; final-size visual inspection confirms the text is unobscured.

## Reproduce

```bash
python figures/make_fig1.py --workdir /projects/nar/nar-validation
python figures/make_fig2.py --workdir /projects/nar/nar-validation
```

The scripts write the per-figure CSV before plotting and render exclusively from that table. Existing `results/` files and frozen activation artifacts are opened read-only.
