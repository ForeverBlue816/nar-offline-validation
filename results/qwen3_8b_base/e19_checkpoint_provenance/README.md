# E19 checkpoint provenance (weights deleted 2026-09-12)

The nine E19 GPTQ checkpoints for Qwen3-8B-Base (three rotations x three weight
protocols, 26 GB each, 233 GB total) were deleted to free space on the project
quota.  They were not uploaded to the Hub: each one takes 17-20 minutes of GPU
time to rebuild (2.7 hours for all nine), which is not worth 233 GB of storage
or of the remaining private Hub quota.

What is kept here, per checkpoint, is what the run recorded about itself:

- `DONE.json` — fold audit, fp32 algebra control, bf16 reparameterisation drift,
  GPTQ settings, QuaRot commit, hardware and elapsed time
- `gptq_audit.csv` — per-module GPTQ statistics

To rebuild one, run the E19 GPTQ stage with the same rotation and protocol; the
calibrated rotation factors it needs are the 1.6 MB in
`<workdir>/activations/qwen3_8b_base/e14_rotations` (kept on disk), and the base
model re-downloads from `Qwen/Qwen3-8B-Base`.  Every E19 number in the report
comes from `results/qwen3_8b_base/e19_*`, which is unaffected.

Note these are a different protocol set from the Qwen3-8B-Base checkpoints in
the private Hub repo `ForeverBlue/nar-w4a4kv4-qwen3-base`, which are E22's
`g128_asym` rows.
