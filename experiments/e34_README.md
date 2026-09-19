# E34 reproduction and interpretation

This experiment extends the frozen E29 result commit `6251b7f` on the same activation-only protocol. The exact hypotheses and operational decisions were written to `report.md` and `e34_preregistration.md` before any E34 forward; `e34_execution_manifest.json` freezes their hashes, the evaluation source, every reused factor and all old result files. Slurm logs begin with the preregistration and manifest hashes. The requested single final commit contains the preregistration, implementation, results and report together.

## Run

Use the same cached model/dataset assets and Python environment documented in `e29_e33_README.md`. New large replay tensors are stored beneath the existing project artifact directory at `<model>/e34/`; no old tensor or result is modified.

```bash
python -m unittest discover -s tests -p 'test_e34*.py'
python nar/e34_anchor_attribution.py --freeze
sbatch slurm_e34.sh qwen3_4b_base
sbatch slurm_e34.sh llama32_3b
# Only after both e34_DONE.json files exist:
python nar/e34_report.py --report
python nar/verify_e34.py
```

`--freeze` preserves an existing manifest; the runner refuses changed frozen source/input hashes. Per-chunk checkpoints flush before atomic rename, so an interrupted new E34 run can recover completed work. E29's four grouped baseline rows are replayed first and must reproduce every saved fp32 chunk NLL bit for bit before new PPL rows begin. E29 files are never rewritten. Per-token cross-entropy is additionally evaluated with `reduction='none'`; its reduced mean may differ from the original fused mean by floating-point summation order, which is logged rather than used to replace the frozen scalar. A finished run should be treated as immutable; the analysis and verifier need no GPU.

## Files under results/<model>

- `e34_per_sequence.csv`, `e34_summary.csv`: every fixed row/seed/chunk NLL, PPL mean, seed SD, named contrast and paired intervals.
- `e34_per_token_seed0.csv` through `seed2.csv`: one wide row per scored position, carrying all four baseline token NLLs, gain, offset_spec, predictor/target positions and both class labels. Splitting by seed keeps each CSV below GitHub's individual-file size limit.
- `e34_dc_share_by_layer.csv`, `e34_flags.json`, `e34_flagged_tokens.csv`, `e34_token_flags.csv`: frozen layer selection, every input's selection score/class, counts and flag hashes.
- `e34_range_common_input.csv`: each down layer/group/seed/column measured on the same unquantized forward activations. Corresponding asymmetric and symmetric formats share this prequantization value.
- `e34_range_runtime.csv`: each actual Part A quantized forward, including both formats; `e34_range_split.csv` preserves both scopes and explicitly empty anchor/other strata; `e34_range_ratios.csv` gives the fixed H34c comparison.
- `e34_attribution.csv`, `e34_attribution_target_position.csv`, `e34_attribution_per_chunk.csv`: primary predictor-position loss attribution, supplemental true-target indexing and the paired chunk sums used for intervals.
- `e34_masked_summary.csv`, `e34_masked_offset_contrast.json`: masked asymmetric PQ−Had deltas, joint sum and interaction contrast, plus the matched-symmetric offset-specific NLL effect per quantized input.
- `e34_gates.csv`, `e34_rotation_contract.csv`, `e34_replay_audit.csv`, `e34_baseline_reuse.csv`: numerical bounds, column identities, minimal swaps, unchanged factors/signs and exact E29 replay/reuse hashes.
- `e34_token_reduction_audit.csv`: differences between the double-precision mean of exported token losses and E29's frozen fp32 reduction; the independent original-reduction replay must still match exactly.
- `e34_hypotheses.json`, `e34_metadata.json`, `e34_DONE.json`: every fixed decision clause, provenance and measurement completeness.

`experiments/e34_run_manifest.json` records the two Slurm jobs and final scheduler states. `e34_job_log_excerpt.txt` preserves their timestamped pre-registration hashes and progress lines. `e34_final_verification.json` contains the final result hashes, completeness checks and old-result preservation audit.

## Limits fixed before measurement

Moving an anchor while retaining a bijective permutation necessarily displaces an occupied coordinate. The implementation uses a minimal same-group swap; all other residual assignments remain fixed. At k=max every PQ group is an anchor group, so an 'other group' comparison cannot be measured. No k=8 substitute was added after observing results.

BOS is an input at position 0, with no BOS-target NLL. The primary BOS row is the first text token predicted from BOS, consistent with indexing next-token loss by its predictor position. The supplemental actual-target table has zero BOS targets and an unavailable BOS mean. Both positions are retained in every token row. The final input position has no scored continuation, and its flag status is logged separately.

H34d requires symmetric masked controls in addition to the requested asymmetric masks: asymmetric-only M1/M2 gains cannot isolate an offset-specific contrast. Signed offset shares can be negative or exceed 100%; a nonpositive total cannot establish concentration of a positive offset benefit. Part B means and mask-normalized rates use 64 paired chunk clusters after averaging seeds, while PPL contrasts retain E29's requested 3×64 paired delta-method intervals and the additional three-seed intervals. Ratios are pooled sums, not equally weighted means over nonempty classes; empty chunks remain in the cluster calculation.

The H34b ratio is the requested target-column intervention ratio. It is not a pure additive causal fraction because the minimal residual swap, group distributions and autoregressive propagation also matter. The report applies the specified minor/two-part/major wording separately to each model and preserves disagreement among the hypothesis checks.
