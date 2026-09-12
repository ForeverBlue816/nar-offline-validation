# Measured evidence appendix

Core timing: 72/72 independent processes complete (50 runs per process).

Three balanced sessions are required for core conclusions. The expanded workloads have one session only.

## Kernel comparisons

Each ratio is candidate elapsed / baseline elapsed. A value above one means a longer elapsed time.


## Decode wall time and memory

- 3b: resident NAR factors 22020096 bytes, shared H128 32768 bytes, removed Hadamard buffer 0 bytes. Predicted loaded delta 22052864 bytes; observed 22052864 bytes; unassigned remainder 0 bytes. Decode partial workspace 131200 bytes; warmed allocated minus uniquely counted storage delta leaves 384 bytes explicitly unassigned. Every session/phase is retained in [memory deltas](memory_deltas.json).
- 3b: NAR vs Hadamard decode wall overhead -2.37%; near parity / within observed variability.
- 8b: resident NAR factors 44040192 bytes, shared H128 32768 bytes, removed Hadamard buffer 50176 bytes. Predicted loaded delta 44022784 bytes; observed 44007424 bytes; unassigned remainder -15360 bytes. Decode partial workspace 131200 bytes; warmed allocated minus uniquely counted storage delta leaves -14976 bytes explicitly unassigned. Every session/phase is retained in [memory deltas](memory_deltas.json).
- 8b: NAR vs Hadamard decode wall overhead -4.36%; consistent observed direction across three sessions, specific to this implementation/workload.

## Allocator alignment check

The native allocator rounding model in [PyTorch2.4.1](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/c10/cuda/CUDACachingAllocator.cpp) rounds ordinary requested blocks to multiples of512 bytes. [The derived audit](allocator_alignment_audit.json) applies this formula separately to every recorded unique storage; it does not alter measured allocated bytes or the unrounded residuals.

- 3b: rounded storage differences exactly match both loaded and warmed NAR-minus-Hadamard allocated differences for 12/12 recorded phase/session pairs. This is evidence consistent with allocator alignment, not an allocator block snapshot or a complete account of external allocations.
- 8b: rounded storage differences exactly match both loaded and warmed NAR-minus-Hadamard allocated differences for 12/12 recorded phase/session pairs. This is evidence consistent with allocator alignment, not an allocator block snapshot or a complete account of external allocations.

## Explicit failures and limits

The operator boundary test is a deployment limitation even if random-weight forwards remain finite. Failed graph capture/replay cannot establish a graph speedup. Real FP16 base-checkpoint consistency cannot establish real INT4 model quality.

Full details: [completion and failures](completion_audit.json), [profiles](profile_summary.json), [kernel ratios and byte estimates](kernel_comparisons.json), [session metrics](metrics_summary.json).
