# E28-v2

Run only in the existing E28 environment on an allocated NVIDIA A40. Historical E28/E17/E22/E26 files are read only. Code lives here; all experiment output belongs under `results/e28_v2/<run_id>/`.

```bash
E28_RUN="$PWD/results/e28_v2/<run_id>"
bash quarot-llama3/e28_v2/run.sh audit --run "$E28_RUN"
bash quarot-llama3/e28_v2/run.sh verify --run "$E28_RUN" --level operators
bash quarot-llama3/e28_v2/run.sh verify --run "$E28_RUN" --level r4 --model 3b
bash quarot-llama3/e28_v2/run.sh verify --run "$E28_RUN" --level model --model 3b --method nar
bash quarot-llama3/e28_v2/run.sh benchmark --run "$E28_RUN" --model 3b --method nar --phase decode --mode eager_sequence --session 1
bash quarot-llama3/e28_v2/run.sh profile --run "$E28_RUN" --model 3b --method nar
bash quarot-llama3/e28_v2/run.sh collect --run "$E28_RUN"
```

`slurm.sh` allocates one A40 for all phases. The pipeline uses new processes for each model/method/mode/phase, three balanced core sessions, and separate profile/kernel/graph jobs within that allocation. `graph` is bounded feasibility/correctness; it cannot create speedup claims without validated comparable timing. `kernel_bench` executes the two fixed ablations and three distinct contracts without autotuning. Extra decode workloads are limited to b8/p2048 and b1/p8192, one50-run session each.

`audit` freezes counts, thresholds and source hashes. `verification_harness_corrections.json` and `correctness/attempt_1/` preserve fixes to the initial verifier without changing thresholds. The execution source manifest identifies corrected code before the main run. Do not edit measurement code during a running formal phase; use a new manifest/run for a computational change.

Completed JSON files are skipped, allowing collection/report changes without GPU reruns. Interrupted/failed files are retained; explicitly move them into a numbered attempt directory before a justified retry. Do not silently overwrite failures. `collect` can run on a CPU/login node and generates JSON summaries, four table types, paper LaTeX and `report_e28_v2.md`.

The benchmark initializes random weights by name and hashes every common state tensor. Real-text inputs validate implementation behavior only. No full PPL/task quality claim, no complete real k8 checkpoint, and no paper-native group128 asymmetric integer deployment are implied. Full cache content/flags/metadata are reset and the prefix rebuilt outside each timing interval. Main decode is120 sequential causal calls following eight warm steps, not a parallel prefill or serving latency.

A separate optional P2 check loads the existing base safetensors into the FP16 backend only:

```bash
bash quarot-llama3/e28_v2/run.sh checkpoint --run "$E28_RUN"
bash quarot-llama3/e28_v2/run.sh verify --run "$E28_RUN" --level model --model 3b --method fp16 --real
```

The real FP16 result has its own `model_real_*` record. It does not establish a compensated k8 INT4 checkpoint or model accuracy. `base_checkpoint_manifest.json` hashes the actual weight files. Cold versus warmed scripted KV packing can differ even on the original wrapper; `cache_diagnose.json` and `verification_warmup_correction.json` retain the evidence and preparation fix.

`collect` also creates `completion_audit.json`, `kernel_comparisons.json`, `memory_deltas.json`, `profile_summary.json` and `evidence_appendix.md`. Ratios are candidate elapsed / baseline elapsed (greater than one means slower); invalid kernel rows are excluded and fewer than three sessions cannot establish a stable benefit. `tables/deployment_sessions.*` preserves run dispersion separately from dispersion across session medians.


The optional bounded graph adapter is built by `build_stream.sh` in an independent CPU allocation. It uses the same pinned QuaRot/CUTLASS sources and CUDA12.4 compiler, changing nine stream arguments across three CUDA files. `local_build/` is an ignored, private build tree; no packages or shared binaries are installed. The exact patch, setup script, compiler log and binary SHA are retained outside that tree. The first CPU launch failed before compilation because of the cluster Lmod path; its log/script remain preserved.

`stream_pipeline --wait-for-primary` waits without initializing CUDA, then runs after the primary pipeline. It first fingerprints actual original/patched operator outputs and mutated KV storage, including the known accumulator-overflow cases. Only exact agreement allows growing-cache graph checks. All three methods must pass before the same private binary enters matched eager/graph timing. These records live in `raw_graph_runs/` and `graph_metrics_summary.json`, separate from original-binary core records; the tables never pool them.

LaTeX output requires `booktabs` and `longtable`. The compact kernel table pools repetitions while retaining three-session counts; `kernel_sessions.*` and raw JSON preserve the individual sessions. No LaTeX compiler is installed in this execution environment, so rendered manuscript integration is not validated here.


For a fresh reproduction, use a new run directory and run `audit` first; published directories preserve their original measurements. The private build creates its stream protocol addendum before compilation if it is absent. Submit the CPU build with `E28_CODE_ROOT="$PWD" E28_RUN="$E28_RUN" sbatch quarot-llama3/e28_v2/build_stream.sh`. Set `E28_STREAM_GRAPH=1` when submitting the main Slurm script to append the gated stream stage after the primary pipeline; the build must be available by then. A missing/failed build is recorded as BLOCKED. Do not reuse an ignored binary path from a cloned published manifest as if the binary had been distributed.

For the recorded `20260912_a40_v2_full_int4` run, the primary and private graph stages use two sequential Slurm allocations on the same asserted GPU UUID. `protocol_graph_allocation_addendum.json` records this scheduling change before any private graph measurements. `slurm_stream.sh` requires the primary completion marker, checks the exact A40 UUID, and places the complete matched private eager/graph panel in the second allocation. Original and private backend measurements are never pooled.

`protocol_kernel_output_addendum.json` freezes native packed-output reconstruction checks before the first microbenchmark, including finite scales/offset reconstruction and the existing 0.002 relative-L2 bound against its matched quantized reference. Dispatch candidates both require byte-equal output. `protocol_rope_reference_addendum.json` freezes a separate scalar-FP64 RoPE check at ten positions, including the 8192-token extension, before real FP16 validation. It does not change model computation. The original and clarified quantizer contracts are both retained with source hashes.

Profiler timelines are complete Chrome JSON traces compressed losslessly as `profiles/*.trace.json.gz` using the installed PyTorch exporter. Each profile summary records both compressed and uncompressed SHA256 and byte counts. Decompress with `gzip -dk path/to/model_method.trace.json.gz` before opening in a viewer that requires plain JSON. No trace events are discarded.

The recorded run recovered from termination of an interactive continuation using `slurm_resume_controller.sh`: a CPU-only batch controller launches the remaining pipeline as a step in the still-active A40 allocation. Its pinned CPU mask is specific to this recorded allocation. The interrupted record and session gap are preserved, completed phases are skipped, and resumed source manifests never overwrite the initial execution manifest. Consult `session_interruption_1.json` and `recovery_allocation_audit.json` for the actual recovery resources and inherited environment-field caveat.
