# E29–E33 preregistration

Recorded UTC: 2026-09-18T04:50:26.783323+00:00

# Ablation prompts E29–E33 (reviewer-driven)

Shared preamble — paste this once at the top of each prompt.

```
Context
- Repo: nar-offline-validation. Follow the frozen activation-only protocol of E20/E27:
  fake-quant only at the post-RMSNorm q/k/v inputs and the down_proj inputs,
  bf16 weights and KV, exact-transpose fold, fp32 per-chunk NLL, the frozen 64
  WikiText-2 test chunks of 2048 tokens, three paired rotation seeds (0,1,2),
  round-trip residual <= 1e-6 per (row, seed, site). Default quantizer:
  asymmetric per-group INT4, g=128, fp16 scale + fp16 zero (4.25 bits/value).
- Primary model: Qwen3-4B-Base (hidden 2560 = 20 slots, intermediate 9728 = 76 slots;
  Paley-block Hadamards as in E22). Secondary model, only where the prompt says so:
  Llama-3.2-3B, so the result can be set beside the existing E11/E20/E27 rows.
- Baselines in every experiment: Hadamard (no alignment) and PrismQuant k=max,
  same seeds, same chunks. Report every row as mean PPL over 3 seeds (6 decimals),
  seed std, and the paired delta to the named reference with a 90% Student-t CI
  over the 3 x 64 paired chunks.
- Pre-registration: write the hypothesis block of the prompt into report.md BEFORE
  the first run. No row is added, dropped, or re-tuned after seeing numbers.
- Never modify or re-run existing result files. New results go under
  results/<model>/e<NN>_*.{csv,json}; per-sequence CSVs are append-merged as in E20.
- Report back in report.md (new "## E<NN>" section with the pre-registration, a
  results table, and a 3–5 sentence reading) and in your reply, with one commit
  per experiment titled "E<NN>: <title> (<model>)".
```

---

## E29 — Symmetric-quantizer control (does the gain need the offset?)

```
Task: test whether PrismQuant's gain depends on the affine offset, as the method claims.

Rows (activation-only, Qwen3-4B-Base and Llama-3.2-3B, both sites quantized):
  Q1  asymmetric g=128 (default)             Hadamard | PrismQuant k=max
  Q2  symmetric   g=128 (scale only,
      absmax/7, zero fixed at 0)             Hadamard | PrismQuant k=max
  Q3  asymmetric per-token (g = d, one
      scale + one zero per token)            Hadamard | PrismQuant k=1 (the only slot)
  Q4  symmetric per-token (g = d)            Hadamard | PrismQuant k=1
Keep the rotation identical across Q1/Q2 (same factors, same seeds); only the
quantizer changes. For Q3/Q4 the rank is capped at 1 by construction — record
the cap in the row metadata.

Pre-registered hypotheses (write before running):
  H29a: PQ − Hadamard delta under Q2 is at most one quarter of the delta under Q1
        on both models (gain largely vanishes without an offset).
  H29b: under Q3 the delta shrinks toward the k=1 value of the rank sweep
        (E11 / Table 3), i.e. the number of free directions, not the offset
        alone, sets the gain.
  H29c: under Q4 PQ and Hadamard are within one 90% CI of each other.

Gates: round-trip residual <= 1e-6; Q1 reproduces the existing g=128 rows for
Llama-3.2-3B within seed noise (report the diff); per-row effective bits
(4.25 / 4.125 / 4+32/d / 4+16/d) logged.

Deliverable table: rows Q1–Q4 x {Hadamard, PQ, delta [CI]} x {Qwen3-4B, Llama-3.2-3B}.
```

---

## E30 — Calibration robustness (size and corpus)

```
Task: measure how much of the PrismQuant gain survives smaller and out-of-domain
calibration sets. Everything below is activation-only, Qwen3-4B-Base only,
PrismQuant k=max vs Hadamard (Hadamard needs no calibration; run it once).

Part A — calibration size, WikiText-2 train split:
  N = 64, 128 (current), 256 sequences of 2048 tokens. Same 3 rotation seeds;
  the sequence subset for each N is drawn with the seed (N=128 must reproduce
  the existing subset for seed 0).
Part B — calibration corpus, N = 128:
  C4 (first validation shard), and one non-web corpus if available in the
  environment (e.g. a code or arXiv shard); evaluation stays on the frozen
  WikiText-2 chunks AND on 64 C4 chunks so every row has an in-domain and an
  out-of-domain number.
Part C — the greedy permutation under shift: for the C4-calibrated rotation,
  additionally record the per-site captured fraction f and the fraction of
  anchor/filler slots that change relative to the WikiText-2-calibrated
  rotation (Jaccard over slot assignments).

Pre-registered hypotheses:
  H30a: PQ − Hadamard on WikiText-2 changes by less than 20% of its N=128 value
        between N=64 and N=256.
  H30b: C4-calibrated PQ retains at least 75% of the WikiText-2-calibrated gain
        when evaluated on WikiText-2, and exceeds it when evaluated on C4.
  H30c: slot assignments differ between corpora, but f differs by less than
        0.03 per site — i.e. the greedy step moves the assignment, not the energy.

Gates: round-trip <= 1e-6; N=128 / WikiText-2 row reproduces the existing
Qwen3-4B row within seed noise.

Deliverable tables: (A) N x {PQ PPL, delta vs Hadamard [CI]}; (B) corpus x eval
set x {PQ PPL, delta [CI]}; (C) per site: f_WT2, f_C4, slot Jaccard.
```

---

## E31 — Anchor placement and residual balancing

```
Task: isolate the contribution of the signed permutation step.

Rows (activation-only, Qwen3-4B-Base, PrismQuant at k=8 and k=max, 3 seeds):
  P1  energy-balanced greedy Pi with low-energy fillers (default)
  P2  identity Pi (anchors at t_i = 1+(i-1)g, all other coordinates in natural order)
  P3  random Pi (seeded; anchors fixed, remaining coordinates shuffled)
  P4  energy-balanced Pi WITHOUT low-energy fillers (unused constant slots take
      the next coordinates in sorted order)
  P5  P1 with the sign vector D fixed to +1 (no random signs)
All rows share G (same Householder factors) so only Pi / D differ; verify
R v_i = ±u_i still holds for every selected direction (log the anchor residual).

Pre-registered hypotheses:
  H31a: at k=max, P1 <= P2 and P1 <= P3 on PPL; the P1−P2 delta is the cost of
        balancing and is reported whatever its sign.
  H31b: at k=8, P4 is worse than P1 (fillers matter only when slots are unused),
        and at k=max P4 = P1 within seed noise (no unused slots).
  H31c: P5 differs from P1 by less than one seed std — random signs are a
        symmetry-breaking device, not a source of gain.

Gates: anchor residual <= 1e-6 for all rows; round-trip <= 1e-6.

Deliverable table: P1–P5 x {k=8, k=max} x {PPL, delta vs P1 [CI]}, plus for each
row the per-group residual-energy spread (max/median over groups at the down site).
```

---

## E32 — Eigenspace estimation error

```
Task: connect eigensolver accuracy to perplexity.

Rows (activation-only, Qwen3-4B-Base, PrismQuant k=max, 3 seeds; Llama-3.2-3B
for the exact row only, since d=3072 makes exact eigendecomposition trivial):
  S1  randomized subspace iteration, 1 pass
  S2  2 passes
  S3  3 passes (default)
  S4  exact eigendecomposition of the explicit second moment (form Sigma in fp64)
For each row and site log: Ritz residual ||Sigma v − lambda v|| / lambda for
the top-k, principal angles between the S_i subspace and the S4 subspace,
captured fraction f, and PPL.

Pre-registered hypotheses:
  H32a: S3 and S4 differ by less than one seed std in PPL on both models.
  H32b: S1 shows the largest principal angle to S4 and the only PPL delta whose
        90% CI excludes zero, if any does.
  H32c: f is monotone in passes (S1 <= S2 <= S3 <= S4) at every site.

Gates: S4 Ritz residual <= 1e-10 (it is exact); round-trip <= 1e-6.

Deliverable table: S1–S4 x {Ritz residual (median over sites), max principal
angle to S4, f at the down site, PPL, delta vs S4 [CI]}.
```

---

## E33 — Quantitative check of the range law (no GPU needed if steps were logged)

```
Task: turn the pooled R^2 into a per-site prediction error.

Data: for every (model, site, k, seed) already in the E11/E20/E22 diagnostics,
the measured mean quantization step s_meas, the paired Hadamard step s_H(g),
and the captured fraction f_k. If s_meas is not logged, recompute it from the
saved rotations on the 64 chunks (activation-only, no quantizer needed).

Compute per site: s_pred = s_H(g) * sqrt(1 − f_k); relative error
e = (s_pred − s_meas) / s_meas; 90% CI of e over the 3 seeds.

Report:
  (1) a predicted-vs-measured scatter per model with the identity line and
      seed-CI whiskers, colored by site type (qkv / down) and by k;
  (2) a table of median |e| and 90th-percentile |e| by model x site type x k;
  (3) the sign of the systematic error, if any, and which of the named
      assumptions (incidental reference capture, unused slots, correlations,
      tails) is consistent with it — one paragraph, no new fitting.

Pre-registered hypothesis:
  H33: median |e| < 5% at the down site for k=max on all three models; the
       q/k/v site, whose Hadamard gap is small, may show larger relative error.

Do not fit any coefficient. Write the figure to figures/fig_rangelaw_persite.pdf
and the table to results/e33_rangelaw_persite.csv; add an "## E33" section to
report.md.
```

## Operational definitions fixed before any new measurement

- Rotation seeds are indices 0,1,2; signs use the existing E20 mapping with base seed 20260902. Calibration/eigensolver randomness and token hashes are recorded separately.
- Existing result files are immutable. Newly requested evaluations of baseline methods create only E29–E33 rows. Identical new baseline measurements may be referenced across these experiments by hash; they are not repeatedly recomputed.
- Mean PPL is the mean of exp(mean fp32 chunk NLL) over the three seeds. The requested 3×64 paired-chunk 90% Student-t interval is computed on the paired delta-method PPL influence values, centered at the mean seed PPL difference (df=191); paired NLL and seed-level PPL intervals (df=2) are also reported. Repeated chunks across seeds are not claimed to be 192 independent texts.
- H29a is interpreted as gain attenuation: with gain=Hadamard−PQ, the Q2 gain is <=0.25 times the positive Q1 gain. The literal signed PQ−Hadamard deltas are always shown. Both quantizers retain the same rotations within Q1/Q2 and Q3/Q4.
- Per-token PQ uses one full-width DC target (rank cap=1); Qwen full-width Hadamards use the existing normalized Paley product, including its actual transpose.
- E30 seed-0 WT2 N=128 is byte-identical to the existing contiguous calibration token cache. Its N=64 is the first half and N=256 extends the frozen stream. Seeds 1/2 draw deterministic nested subsets of this WT2 training stream. C4 calibration and evaluation use disjoint document/token windows from the first validation shard, with overlap checks. No local code/arXiv corpus was found at preregistration; the explicitly optional non-web row is marked unavailable rather than replaced with a different corpus after seeing results. All E30 PQ rows and the shared Hadamard baseline are evaluated on both fixed evaluation sets.
- Existing Qwen3-4B E22 PPL is W4A4KV4, not the requested activation-only two-site protocol. It cannot be a numerical replication target. The new WT2 N=128 activation-only reference is established here; frozen factors and subsets are audited separately. Llama activation-only baseline differences are reported against compatible E20/E27 records with their protocol differences explicit.
- E31 fixes G within each rank. Identity leaves every selected anchor at i*g; random permutes only non-anchor coordinates. P4 fills unused DC slots by increasing coordinate index before applying the same greedy energy balancing; at k=max P4 is identically P1. Jaccard is over (source coordinate,target coordinate) assignments, separately for anchors/fillers and all coordinates.
- E32 forms the full uncentered second moment from bf16 calibration activations with fp64 products/accumulation. All solver rows use this same matrix. Pass p means p covariance products including the final Rayleigh–Ritz product, matching the existing three-product default. Oversampling is 16; starting sketch and calibration are fixed across solver rows, rotation signs vary over three seeds. Exact eigh is fp64; all top-k Ritz residuals must be <=1e-10. The Llama S3 comparison is the shared default activation-only baseline; only S4 adds a new Llama solver row, as requested. Numerical reorthogonalization/Householder construction in fp64 is permitted to meet the stated 1e-6 gates and is logged, without changing selected subspaces.
- E33 starts with every recoverable E11/E20/E22 diagnostic row, preserving source model/site/layer/k/group/seed and reporting missing seed counts. Existing aggregated or one-seed rows cannot supply a three-seed CI. The primary new matched three-seed check covers Llama-3.2-3B, Llama-3.1-8B and Qwen3-4B-Base on all 64 frozen evaluation chunks, with saved rotations and no quantization. No coefficient is fitted; model-specific rank grids and any unavailable source rotations are listed before measurement. Calibration and test-set energy fractions are distinguished.
- Failures of hypotheses remain results. A numerical gate failure blocks that row and is preserved; no failed measurements are silently replaced. New artifacts are resumable and append-merged using (experiment,model,row,seed,eval_set,chunk) keys.
