# E28-v2 kernel implementation

For row vectors, the actual online formula is

`U = X A; Z = X H_block - U B`,

where X has shape T×d, A is the exported `y_prime_fp32` (d×8), and B is `w_h_t_fp32` (8×d). H_block is normalized Hadamard128 applied independently to channels. Rank8 is the low-rank correction dimension; it does not reduce the model to eight channels.

`export_nar_factors.py` converts the calibrated E11 Householder reflectors to compact WY, applies signed permutation Q=SP to the factors, and forms the block-Hadamard-transformed second factor. The corresponding permutation/sign must be folded into gate/up weights in a legitimate real model. `FoldedR4.q_unfolded` maps captured original down inputs to this online coordinate system; the all-layer check compares the original reflector path to the actually exported factors. The benchmark itself uses random weights and makes no claim of real offline compensation or quality equivalence.

## Kernel A and B

Kernel A reuses `r4_fused_v3.rank_projection_dot_kernel`. A token tile and channel tile iterate over a fixed channel split; masks bound both the split and the actual dimension. The d×8 factor is represented by FP16 high and low terms, each padded to16 columns. Each term gets its own FP32 dot-product accumulator; they combine only at the end. A writes a real FP32 array `[T, splits*8]`, not a d×d rotation.

Kernel B owns a token tile and one128-channel block. The TC version reads a shared FP16 128×128 ±1 matrix, accumulates `X_tile @ H128` in FP32, then normalizes by `1/sqrt(128)`. It sums A's split partials and splits the FP32 U into high/low FP16 terms. The correction uses high×high + low×high + high×low; low×low is omitted. Error is measured for every layer and actual launch shape using the frozen relative-L2 and actual-quantizer code-match thresholds. A theoretical small omitted term is not a substitute for those checks.

The two-launch structure avoids a full-row reduction and rank correction inside every group program. It costs one scratch write/read and a second launch. The generic and prebound launch paths execute the same compiled arithmetic; prebinding removes repeated Python/Triton argument binding. Prebound availability and byte-equal output checks are recorded, rather than assumed.

The shuffle B ablation reuses `_fwht128` and the FP32 rank-correction helper, with the same FP16 input/output interface. It is not E17's quantization epilogue. E28 B writes FP16, then the unchanged QuaRot quantizer writes actual packed codes and per-token scales. E17 B instead directly writes group128 asymmetric packed codes/scales/offsets. Full-width QuaRot Hadamard and E17 block-Hadamard are different baselines, so their rows remain separate.

## Fixed configurations and storage

Exact selected A/B configurations for each model and T=1,2048,32768 are copied into `protocol.json`. No new search is run. The only shuffle ablation uses B tile8,4warps,2stages; E17 native B uses that same fixed tile. Layer0 supplies timing factors; all layers supply correctness checks. Reused hot inputs/factors can benefit from L2 cache.

At k8 and padded rank16, per-layer resident factors are:

- Y high/low: `2*d*16*2 = 64d` bytes.
- W high/low: `2*8*d*2 = 32d` bytes.
- Total: `96d` bytes/layer; d8192 means786432 bytes (768KiB), d14336 means1376256 bytes (1344KiB).
- One H128 across layers: `128*128*2 = 32768` bytes.
- One partial array per distinct `(device,T,splits*k)` shape: `4*T*splits*8` bytes, shared across sequential layers.

The original source FP32 factors are released after constructing the model. Shared H128 and partial arrays are counted once by underlying storage. The scratch lifetime spans the process and assumes sequential calls on the same execution stream; multi-stream concurrent use is not established. Standalone decode is measured in a new process so it cannot inherit b16 scratch.

Arithmetic scales as O(Tdk) for A, O(Tdk) for correction and O(Td*128) for the dense128 TC Hadamard implementation. Rank padding gives the current k≤16 implementation a common tile width; this does not mean arbitrary rank has constant cost.

A compulsory-data traffic estimate for E28 A+B is `6Td + 96d + 8T*splits*k + 32768` bytes: X read by A/B, FP16 Z write, factors, scratch write/read, and shared H. Actual program-level requests reread factors per token tile, scratch per channel group and H per B program; caches can satisfy many requests. Consequently any bytes/time value is an effective-bandwidth estimate, not measured DRAM utilization. No bandwidth-bound or compute-bound claim is established without counters. The directly measured chain is never replaced by a sum of stage medians.
