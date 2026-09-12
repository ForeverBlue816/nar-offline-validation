# E28-v2: auditable deployment experiments

Generated 2026-09-12T09:53:24.758673+00:00. This report contains completed records only where the linked JSON says COMPLETE/PASS; other stages remain explicitly incomplete, failed or blocked. All model performance rows use **random weights**, even when validation inputs are real text. No full model-quality evaluation is claimed.

## Initialization revision and reused evidence

The first E28-v2 cohort used the legacy1..6 packed-byte range with name-stable shared weights: half the channels are zero and all other weights positive. Its8B NAR model produced nonfinite outputs: the last decoder residual overflowed FP16 although the R4 operators stayed finite. [Original failure and partial measurements](../20260912_a40_v2/report_e28_v2.md) are retained. This run uses one predefined uniform0..255 packed-byte initialization for both integer methods, with unchanged scales/seeds/kernels/thresholds. It is a separate frozen random-weight protocol, not a claim of improved model quality.

[Allocator-preflight correction](allocator_preflight_revision.json) identifies the first two affected processes and their fully retained/repeated measurements. The reruns apply one consistent memory-preparation protocol; no run was selected by speed.

[Reused correctness records](reused_correctness.json) identify unchanged local operators/factors and FP16 states verified in the same A40 allocation. All four integer model paths are rechecked for the new state; all formal performance sessions are newly measured.

## One-page status

Core timing: **72/72** independent processes complete. Full timing completion does not override any numerical failure below. The [stage completion table](tables/completion.md) separately tracks every P1/P2 check and timing panel.

- Preserved original E28/E17/E22/E26 artifacts.
- Froze counts, shapes, selected configs, thresholds and exclusions in [protocol.json](protocol.json); execution provenance in [source_manifest.json](source_manifest.json) and [execution_source_manifest.json](execution_source_manifest.json).
- Corrected shared INT4 initialization, stale RoPE caches, timing boundaries, complete cache reset, throughput statistics, independent memory and invalid acceleration-retention ratios. The common metadata optimization is checked against the original wrapper with the same rebuilt RoPE caches on both sides; this does not establish equivalence to historical stale-RoPE outputs.
- The exact original environment and A40 are reused; no new GEMM, calibration or serving framework.
- Native paper-format integer deployment is BLOCKED by missing matching complete checkpoint and incompatible group-scale/KV interfaces.

| Check | Status | Reason |
| --- | --- | --- |
| cache_diagnose | DIAGNOSTIC_RECORDED |  |
| model_3b_fp16 | PASS |  |
| model_3b_hadamard | PASS |  |
| model_3b_nar | PASS |  |
| model_8b_fp16 | PASS |  |
| model_8b_hadamard | PASS |  |
| model_8b_nar | PASS |  |
| operators_all | FAIL |  |
| r4_3b | PASS |  |
| r4_8b | PASS |  |

## Numerical boundaries and real weights

- `int4_gemm_dequant` / large_accumulator, reduction width 8192: integer accumulation 401408 is exact, but conversion to FP16 before scale multiplication produces nonfinite output. This remains **FAIL** in the existing backend.
- `int4_gemm_dequant` / large_accumulator, reduction width 14336: integer accumulation 702464 is exact, but conversion to FP16 before scale multiplication produces nonfinite output. This remains **FAIL** in the existing backend.

Real base-checkpoint FP16 implementation checks: 3b: PENDING; 8b: PENDING. The independent pinned RoPE reference is recorded in `correctness/rope_reference_3b.json` and `correctness/rope_reference_8b.json` when run. This is separate from compensated INT4 checkpoint support and from model quality. Prefill hidden/logits and multi-step decode are checked; exact cache comparisons cover steps1,2,9,64,65,128.

## Six questions

1. **Which stages accelerate relative to FP16?** Ratios above1x indicate faster execution than FP16; ratios below1x indicate slower execution. Same-mode ratios only; [raw metrics and session status](metrics_summary.json), [deployment prefill](tables/deployment_prefill.md), [decode](tables/deployment_decode.md).
- 3b prefill1: Hadamard/FP16 speedup 1.26x; PrismQuant/FP16 1.25x. hadamard: faster than FP16 in all three sessions; nar: faster than FP16 in all three sessions.
- 3b prefill16: Hadamard/FP16 speedup 1.28x; PrismQuant/FP16 1.26x. hadamard: faster than FP16 in all three sessions; nar: faster than FP16 in all three sessions.
- 3b decode: Hadamard/FP16 speedup 0.51x; PrismQuant/FP16 0.52x. hadamard: slower than FP16 in all three sessions; nar: slower than FP16 in all three sessions.
- 8b prefill1: Hadamard/FP16 speedup 1.45x; PrismQuant/FP16 1.46x. hadamard: faster than FP16 in all three sessions; nar: faster than FP16 in all three sessions.
- 8b prefill16: Hadamard/FP16 speedup 1.49x; PrismQuant/FP16 1.51x. hadamard: faster than FP16 in all three sessions; nar: faster than FP16 in all three sessions.
- 8b decode: Hadamard/FP16 speedup 0.57x; PrismQuant/FP16 0.60x. hadamard: slower than FP16 in all three sessions; nar: slower than FP16 in all three sessions.

2. **What does PrismQuant add versus Hadamard?** See directly measured A/B/quantizer/chain times in [kernel table](tables/kernel.md), same-mode wall ratios in [metrics](metrics_summary.json), and exact unique-storage categories/deltas in [memory_breakdown.json](memory_breakdown.json) and [memory_deltas.json](memory_deltas.json). Factor storage is96d bytes per layer plus one shared32768-byte H128; scratch is recorded separately. No rounded-to-zero overhead claim.
- 3b prefill1: NAR throughput is 98.87% of Hadamard; consistent observed direction across three sessions, specific to this implementation/workload.
- 3b prefill16: NAR throughput is 98.52% of Hadamard; consistent observed direction across three sessions, specific to this implementation/workload.
- 3b decode: NAR latency difference -2.37%; near parity / within observed variability.
- 8b prefill1: NAR throughput is 100.92% of Hadamard; near parity / within observed variability.
- 8b prefill16: NAR throughput is 101.11% of Hadamard; consistent observed direction across three sessions, specific to this implementation/workload.
- 8b decode: NAR latency difference -4.36%; consistent observed direction across three sessions, specific to this implementation/workload.

3. **Does dispatch optimization help reproducibly?** The generic/prebound protocol requires identical kernels, byte-equal outputs and three separately recorded sessions; only completed valid records support a conclusion. See [kernel records](tables/kernel.md). A one-percent difference without consistent session direction is near parity, not a stable advantage.

4. **Does graph replay grow the KV context?** See [graphability audit](graphability_audit.md) and the graph correctness rows above. Only A-B-A replay with steps1,2,9,64,65,128, full cache hashes and actual page crossing can pass. Original-binary capture and the bounded private current-stream adapter have separate records. Any matched adapter timing is in the [separate graph panel](tables/graph_deployment.md), with [preparation and memory](graph_memory_breakdown.json). No graph performance panel is populated from a fixed-context microbenchmark; absent timings remain null.

5. **Which results execute integer kernels?** E28 Hadamard/NAR use the actual packed INT4 GEMM/quantizer and INT4 decode cache; FP16 uses FP16 arithmetic/cache. E17-native rows are local packed-activation microbenchmarks, not an end-to-end model. Random states are verified by [shared_state_audit.json](shared_state_audit.json). Real model quality remains untested.

6. **What is missing for the paper's group128 asymmetric path?** See [quantizer contract](quantizer_contract.md) and [checkpoint audit](checkpoint_audit.json). Per-group reduction scales/offsets need partial sums/corrections absent from this GEMM interface; token/channel KV grouping and residual windows also differ. No discarded offsets or FP16 substitute is presented as native integer deployment.

## Session interruption and durable recovery

The interactive driver step150777.7 was terminated during session2. Its interrupted prefill16 process had no complete timing samples; all29 complete processes were retained. The interrupted JSON is preserved under `raw_runs/interrupted_attempt_1/`, and only that unfinished process restarts with the same10 warmups and50 formal runs. The unplanned idle gap within session2 remains a protocol deviation, not a reason to select or drop timings. See [interruption record](session_interruption_1.json).

A durable CPU Slurm controller launches the continuation within the original A40 allocation. [Actual step/cgroup audit](recovery_allocation_audit.json) confirms the same cores32--35, no CFS quota,40GiB host-memory limit and original GPU node. Raw environments preserve some inherited controller job fields; the actual GPU step TRES, host, affinity and cgroups are authoritative. Original execution manifests remain intact; resumed code has a separate timestamped manifest.

## Technical appendix

[Kernel design](kernel_design.md), [old claims audit](old_claim_audit.json), [compatibility table](tables/compatibility.md), [memory table](tables/memory.md), and [measured evidence appendix](evidence_appendix.md), [profile summary](profile_summary.json) and [profile traces](profiles/) provide the detailed evidence. The [predecessor attempt directory](../20260912_a40_v2/correctness/attempt_1/) retains initial harness failures; [harness corrections](verification_harness_corrections.json) explain the repair without changing numerical thresholds.

All tables are regenerated by `collect`, which never launches a GPU experiment. Memory is bytes/decimal GB; pooled150 runs represent3 sessions. Profiler kernel sums, CUDA-event elapsed and wall time are distinct. Missing counters mean bandwidth/compute bottleneck labels remain hypotheses.
