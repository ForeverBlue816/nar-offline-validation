# E29–E33 reproducibility

The hypotheses were committed in `00d28a0` before measurements. `e29_e33_preregistration.md` preserves the complete request and operational definitions. Subsequent numerical and execution amendments are explicitly recorded alongside it; the measured row matrix is fixed.

## Frozen protocol

Only post-RMSNorm qkv inputs and down-projection inputs are fake-quantized. Original weights and KV remain bf16. The activation transform uses the actual transpose, followed by the original bf16 linear layer; chunk cross-entropy is computed in fp32. Full frozen 64 × 2048-token evaluation tensors are shared by every paired row. Calibration size/corpus changes belong only to E30. Qwen's first 64 test chunks are an exact prefix of the existing 146-chunk frozen tensor. Rotation seed indices are 0, 1 and 2 with the E27 +128 sign mapping; legacy E20 range rows retain their original mapping.

The default quantizer stores fp16 scale and real-valued offset at g=128. The symmetric control uses absmax/7 and a zero offset. Effective bits are logged separately for each site width. Both quantizers in each E29 pair share precisely the same rotations. The full-width PQ row has one slot and a DC-normalized Paley matrix; the normalization is a fixed matrix property, recorded before any E29 PPL. Grouped transforms and the frozen Hadamard baseline are unchanged.

Householder construction and eigensolver accumulation use fp64. For actual PQ evaluations, the two compact-WY products accumulate in fp64 using the unchanged stored fp32 factors; Hadamard and quantization use fp32. This adjustment followed an anchor-gate failure of 1.0326e-6 before the first PQ PPL chunk. The preflight shows that fp64 multiplication reduces the maximum anchor error to 1.0857e-7 without changing factors or subspaces. All formal rows still undergo the requested numerical gates.

Round-trip checks apply the actual forward and transpose to eight fixed Gaussian probe vectors at every layer/site, reporting relative Frobenius error. Anchor checks evaluate every selected unit direction and report the largest Euclidean distance to its signed constant-slot target. The final audit checks every recorded row/seed/layer/site against the 1e-6 threshold.

## Execution

Use the repository's existing CUDA/PyTorch environment and cached model/dataset assets. `reviewer_ablations.py` contains the project cache defaults; `ABLATION_ASSETS` may override the new artifact destination. Old result files and model assets are read only.

```bash
python nar/reviewer_layerwise.py --model qwen3_4b_base --exact
python nar/reviewer_layerwise.py --model llama32_3b --exact
python nar/run_reviewer_experiments.py --experiment 29 --model qwen3_4b_base
python nar/run_reviewer_experiments.py --experiment 29 --model llama32_3b
python nar/run_reviewer_experiments.py --experiment 31 --model qwen3_4b_base
python nar/run_reviewer_experiments.py --experiment 32 --model qwen3_4b_base
python nar/run_reviewer_experiments.py --experiment 32 --model llama32_3b
python nar/prepare_e30.py
python nar/run_reviewer_experiments.py --experiment 30 --model qwen3_4b_base
python nar/e33_rangelaw.py --model llama32_3b
python nar/e33_rangelaw.py --model llama31_8b
python nar/e33_rangelaw.py --model qwen3_4b_base
```

`prepare_e30.py --seed 0/1/2` can independently prepare each seed, subject to the scheduler's job-count limit. Slurm wrappers retain the environment settings and append logs after requeue. Layerwise calibration is verified bit for bit against complete unquantized model forwards on two sequences, including every qkv/down capture and final hidden state. Each layer consumes the entire requested calibration sample before activations are released; no eigensolver samples are dropped. Only the historical stride-32 energy sample is used for the greedy permutation, matching the frozen algorithm.

New per-chunk CSVs append by unique row/seed/evaluation/chunk keys, with conflicting values rejected. All new torch and CSV writes flush before rename. Interrupted intermediate files are quarantined and reconstructed from frozen token IDs; completed measurements are retained. Identical new baselines are reused across experiments with a hash and row audit. Existing published result files are never rewritten.

## Statistics and diagnostics

Per-seed corpus PPL is exp(mean chunk NLL); the displayed estimate averages the three seed corpus PPLs. Seed SD uses ddof=1. The requested paired 90% chunk interval uses t(191) times the standard error of delta-method PPL influence values, centered on the mean paired seed corpus-PPL difference. All 3 × 64 paired chunks enter the calculation. Repeated texts across seeds are not 192 independent documents; paired NLL intervals and the more conservative three-seed PPL interval are also saved. E33 instead reports t(2) intervals directly over the three seed prediction errors/steps, as requested.

`reviewer_diagnostics.py` exports all E30 layer/seed slot assignments, E31 shared-G checks, and every E32 top-k Ritz residual and principal angle. `reviewer_report.py` checks completion before inserting tables and hypothesis outcomes into each existing pre-registration section. The decision rules for ambiguous “one seed SD” comparisons and the unavailable historical k=1 reference are fixed in `e29_e32_decision_rules.json`.

```bash
python nar/reviewer_diagnostics.py --experiment 30
python nar/reviewer_diagnostics.py --experiment 31
python nar/reviewer_diagnostics.py --experiment 32 --model qwen3_4b_base
python nar/reviewer_diagnostics.py --experiment 32 --model llama32_3b
python nar/e33_rangelaw.py --archive
python nar/reviewer_report.py --experiment 29  # repeat for 30, 31, 32, 33
python figures/plot_e33_rangelaw.py
```

The E33 figure uses every complete new layer/configuration aggregate, identity lines and genuine seed-CI whiskers. Both axes are logarithmic with equal limits and physical scale within each model; hues distinguish sites and shade intensity encodes absolute rank. Single-seed historical range diagnostics are archived separately without invented replication. The rank-pooled error table and the more specific configuration-separated table are both provided; no coefficient is fitted. PDF, editable SVG and 600-dpi PNG accompany rendered alignment, font and collision audits. Source-code QA reads both the plotting script and its shared exporter; its sans-serif-only heuristic is adjudicated against the manuscript's existing Times New Roman contract.

## Final verification and figure dependencies

The recorded Python/package versions are in `e29_e33_software_environment.json`. Figure rendering uses matplotlib, Pillow, the existing repository fonts and PyMuPDF for PDF geometry audits. On the recorded cluster, the added PyMuPDF installation is isolated in `/home/yanlongc/figure-qa-deps`, so the figure command uses `PYTHONPATH=/home/yanlongc/figure-qa-deps`; in another environment, install PyMuPDF in that environment instead.

```bash
python -m unittest discover -s tests -p 'test_reviewer_*.py'
python nar/verify_reviewer_results.py
PYTHONPATH=/home/yanlongc/figure-qa-deps python figures/plot_e33_rangelaw.py
```

The final audit requires every planned chunk and every row/seed/layer/site numerical gate, checks each evaluation tensor hash, verifies the exact-solver residuals and E33 prediction formula, and confirms that the original result files and report prefix are preserved. Its output is `e29_e33_final_verification.json`; `e30_token_audit.json` additionally checks nested calibration subsets, the exact seed-0 baseline, and disjoint C4 documents/windows. Large model, calibration and rotation tensors remain in the project artifact store rather than Git; the source scripts, frozen-input hashes, measured tables, figures and execution provenance are versioned.
