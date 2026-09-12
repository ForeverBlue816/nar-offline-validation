# Measured evidence appendix

Core timing: 11/72 independent processes complete (50 runs per process).

Three balanced sessions are required for core conclusions. The expanded workloads have one session only.

## Kernel comparisons

Each ratio is candidate elapsed / baseline elapsed. A value above one means a longer elapsed time.


## Decode wall time and memory

- 3b: resident NAR factors 22020096 bytes, shared H128 32768 bytes, removed Hadamard buffer 0 bytes. Predicted loaded delta 22052864 bytes; observed 22052864 bytes; unassigned remainder 0 bytes. Decode partial workspace 131200 bytes. Every session/phase is retained in [memory deltas](memory_deltas.json).
- 3b: NAR vs Hadamard decode wall overhead -5.08%; fewer than three comparable sessions; no stable advantage established.

## Explicit failures and limits

The operator boundary test is a deployment limitation even if random-weight forwards remain finite. Failed graph capture/replay cannot establish a graph speedup. Real FP16 base-checkpoint consistency cannot establish real INT4 model quality.

Full details: [completion and failures](completion_audit.json), [profiles](profile_summary.json), [kernel ratios and byte estimates](kernel_comparisons.json), [session metrics](metrics_summary.json).
