# Measured evidence appendix

Core timing: 72/72 independent processes complete (50 runs per process).

Three balanced sessions are required for core conclusions. The expanded workloads have one session only.

## Kernel comparisons

Each ratio is candidate elapsed / baseline elapsed. A value above one means a longer elapsed time.

- 3b T=1 E28 slot: nar_module / hadamard_fp16, wall 0.75, CUDA-event 0.76; near parity / within observed variability.
- 3b T=1 E28 frontend: nar_plus_quantizer / hadamard_plus_quantizer, wall 0.80, CUDA-event 0.82; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=1 Dispatch: nar_prebound / nar_generic, wall 0.53, CUDA-event 0.53; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=1 B implementation: nar_generic / nar_generic_shuffle, wall 1.10, CUDA-event 1.12; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=1 E17 native: nar_native / block_hadamard_native, wall 2.59, CUDA-event 2.61; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=2048 E28 slot: nar_module / hadamard_fp16, wall 1.30, CUDA-event 1.34; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=2048 E28 frontend: nar_plus_quantizer / hadamard_plus_quantizer, wall 1.08, CUDA-event 1.08; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=2048 Dispatch: nar_prebound / nar_generic, wall 0.91, CUDA-event 0.92; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=2048 B implementation: nar_generic / nar_generic_shuffle, wall 0.38, CUDA-event 0.38; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=2048 E17 native: nar_native / block_hadamard_native, wall 1.29, CUDA-event 1.29; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=32768 E28 slot: nar_module / hadamard_fp16, wall 1.54, CUDA-event 1.54; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=32768 E28 frontend: nar_plus_quantizer / hadamard_plus_quantizer, wall 1.17, CUDA-event 1.17; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=32768 Dispatch: nar_prebound / nar_generic, wall 0.99, CUDA-event 0.99; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=32768 B implementation: nar_generic / nar_generic_shuffle, wall 0.39, CUDA-event 0.39; consistent direction in these three sessions; no cross-hardware claim.
- 3b T=32768 E17 native: nar_native / block_hadamard_native, wall 1.77, CUDA-event 1.77; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=1 E28 slot: nar_module / hadamard_fp16, wall 0.38, CUDA-event 0.39; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=1 E28 frontend: nar_plus_quantizer / hadamard_plus_quantizer, wall 0.58, CUDA-event 0.58; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=1 Dispatch: nar_prebound / nar_generic, wall 0.63, CUDA-event 0.58; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=1 B implementation: nar_generic / nar_generic_shuffle, wall 1.05, CUDA-event 1.08; near parity / within observed variability.
- 8b T=1 E17 native: nar_native / block_hadamard_native, wall 1.92, CUDA-event 1.83; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=2048 E28 slot: nar_module / hadamard_fp16, wall 0.73, CUDA-event 0.73; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=2048 E28 frontend: nar_plus_quantizer / hadamard_plus_quantizer, wall 0.85, CUDA-event 0.86; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=2048 Dispatch: nar_prebound / nar_generic, wall 0.96, CUDA-event 0.96; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=2048 B implementation: nar_generic / nar_generic_shuffle, wall 0.33, CUDA-event 0.33; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=2048 E17 native: nar_native / block_hadamard_native, wall 1.27, CUDA-event 1.28; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=32768 E28 slot: nar_module / hadamard_fp16, wall 0.77, CUDA-event 0.77; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=32768 E28 frontend: nar_plus_quantizer / hadamard_plus_quantizer, wall 0.89, CUDA-event 0.89; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=32768 Dispatch: nar_prebound / nar_generic, wall 1.00, CUDA-event 1.00; near parity / within observed variability.
- 8b T=32768 B implementation: nar_generic / nar_generic_shuffle, wall 0.38, CUDA-event 0.38; consistent direction in these three sessions; no cross-hardware claim.
- 8b T=32768 E17 native: nar_native / block_hadamard_native, wall 1.78, CUDA-event 1.78; consistent direction in these three sessions; no cross-hardware claim.

## Decode wall time and memory

- 3b: resident NAR factors 22020096 bytes, shared H128 32768 bytes, removed Hadamard buffer 0 bytes. Predicted loaded delta 22052864 bytes; observed 22052864 bytes; unassigned remainder 0 bytes. Decode partial workspace 131200 bytes; warmed allocated minus uniquely counted storage delta leaves 384 bytes explicitly unassigned. Every session/phase is retained in [memory deltas](memory_deltas.json).
- 3b: NAR vs Hadamard decode wall overhead -2.37%; near parity / within observed variability.
- 3b: separate profiled NAR minus Hadamard kernel sum 0.33 ms/step. This difference is not a wall-time estimate.
- 8b: resident NAR factors 44040192 bytes, shared H128 32768 bytes, removed Hadamard buffer 50176 bytes. Predicted loaded delta 44022784 bytes; observed 44007424 bytes; unassigned remainder -15360 bytes. Decode partial workspace 131200 bytes; warmed allocated minus uniquely counted storage delta leaves -14976 bytes explicitly unassigned. Every session/phase is retained in [memory deltas](memory_deltas.json).
- 8b: NAR vs Hadamard decode wall overhead -4.36%; consistent observed direction across three sessions, specific to this implementation/workload.
- 8b: separate profiled NAR minus Hadamard kernel sum 0.52 ms/step. This difference is not a wall-time estimate.

## Allocator alignment check

The native allocator rounding model in [PyTorch2.4.1](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/c10/cuda/CUDACachingAllocator.cpp) rounds ordinary requested blocks to multiples of512 bytes. [The derived audit](allocator_alignment_audit.json) applies this formula separately to every recorded unique storage; it does not alter measured allocated bytes or the unrounded residuals.

- 3b: rounded storage differences exactly match both loaded and warmed NAR-minus-Hadamard allocated differences for 15/15 recorded phase/session pairs. This is evidence consistent with allocator alignment, not an allocator block snapshot or a complete account of external allocations.
- 8b: rounded storage differences exactly match both loaded and warmed NAR-minus-Hadamard allocated differences for 15/15 recorded phase/session pairs. This is evidence consistent with allocator alignment, not an allocator block snapshot or a complete account of external allocations.

## Explicit failures and limits

The operator boundary test is a deployment limitation even if random-weight forwards remain finite. Failed graph capture/replay cannot establish a graph speedup. Real FP16 base-checkpoint consistency cannot establish real INT4 model quality.

Full details: [completion and failures](completion_audit.json), [profiles](profile_summary.json), [kernel ratios and byte estimates](kernel_comparisons.json), [session metrics](metrics_summary.json).
