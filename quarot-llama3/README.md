# quarot-llama3 — E28 patch directory

Everything E28 (end-to-end throughput and memory, W4A4KV4 vs fp16) adds on top
of QuaRot's released integer pipeline lives here.  QuaRot itself is checked out
unmodified at `spcl/QuaRot@5008669` under `$NAR_WORKDIR/external/quarot`, with
exactly one patch applied (`quarot_gqa_decode.patch`).  Nothing outside this
directory and `results/e28/` is touched.

| file | role |
|---|---|
| `build_env.sh` | builds `fast-hadamard-transform` (QuaRot submodule v1.0.4.post1) and `quarot._CUDA` (CUTLASS 3.4.1 INT4 GEMM, sym_quant/dequant, FlashInfer INT4 KV cache) into the E28 venv with the cluster's CUDA/12.4.0 module |
| `quarot_gqa_decode.patch` | the only change to QuaRot's sources: grouped-query attention in the FlashInfer decode kernel (Llama-3 has 8 KV heads for 24/32 query heads) |
| `modeling_llama3.py` | QuaRot's `e2e/quantized_llama/modeling_llama.py`, ported to Llama-3; every change is marked `# E28:` |
| `nar_r4_kernel.py` | the E17 v3 NAR R4 kernel as a drop-in for `quarot.nn.OnlineHadamard` (row 3) |
| `export_nar_factors.py` | exports the per-layer folded factors Y', W'' from the main environment |
| `e28_bench.py` | the driver: `select` (verified kernel-config selection), `bench` (one row), `collect` (metrics.json, table.tex, env.txt) |
| `run_e28.sh`, `slurm_e28.sh` | environment wrapper and the SLURM chain (three rows x two models, then the E17 v3 A40 microbenchmark) |

## Environment

QuaRot's e2e model was written against transformers 4.36; the repository's last
commit is a dependabot bump of the pin to 4.38.0, under which the released e2e
model does not run (4.38.0 routes every non-static cache through
`DynamicCache.from_legacy_cache`, which needs `len()`, and changed the rotary
call signature).  The E28 venv therefore holds **torch 2.4.1+cu124, transformers
4.36.2, flash-attn 2.6.3 (prebuilt wheel), triton 3.0.0** plus the two
extensions built by `build_env.sh`.  Exact versions and commits: `results/e28/env.txt`.

## Modifications to QuaRot, all of them

### Source patch (`quarot_gqa_decode.patch`, applied to the checkout)

1. `quarot/kernels/include/flashinfer/decode.cuh`: `BatchDecodeWithPagedKVCacheKernel`
   maps query head `h` to KV head `h / (num_qo_heads / num_kv_heads)` when reading
   the paged K/V data and their scales.  Identity for Llama-2 (group size 1).
2. `quarot/kernels/flashinfer.cu`, `include/flashinfer.h`: the decode wrappers take
   `num_qo_heads` and launch one block per (batch, query head) instead of (batch, KV head).
3. `quarot/kernels/bindings.cpp`: `batch_decode_i4/f16` read `num_qo_heads` from
   `q.size(1)` and check that it is a multiple of the KV head count.

### Model code (`modeling_llama3.py`, copied from QuaRot's file; marked `# E28:`)

4. **Config**: transformers 4.36 rejects Llama-3's `rope_scaling: {"rope_type": "llama3", ...}`;
   the config is loaded with it removed and the Llama-3 inverse frequencies
   (`_compute_llama3_parameters` of transformers ≥ 4.43, reproduced verbatim) are
   installed on every layer's rotary embedding.
5. **KV cache head dimension**: `build_cache` used `hidden_size // num_key_value_heads`,
   which is 512 for Llama-3 (GQA); it now uses `hidden_size // num_attention_heads` = 128.
6. **R2 online Hadamard for 24 heads** (Llama-3.2-3B): QuaRot's `OnlineHadamard(num_heads)`
   factors 24 = 12 × 2 and runs `fast_hadamard_transform` on a last dimension of 2
   followed by a broadcast `had_12 @ x` over batch × tokens × head_dim (3M tiny
   matmuls) — 2.2 ms per layer at 2048 tokens on the A40, 30 % of the whole prefill.
   `OnlineHadamardDense` applies the identical H₂₄ = (had₁₂ ⊗ H₂)/√24 as one
   (tokens·128, 24) @ (24, 24) fp16 GEMM, checked against `matmul_hadU_cuda` at
   construction.  Used by rows 2 and 3 alike; 32 heads (Llama-3.1-8B) keep QuaRot's kernel.
7. **Prefill logits**: `LlamaForCausalLM.forward` materialises fp16 logits for every
   position and upcasts them to fp32 (batch 16 × 2048 × 128 256 vocab = 25 GB for
   Llama-3, which does not fit next to the 8B fp16 weights on a 48 GB A40 and would
   dominate every row's peak memory).  With `logits_last_only = True` (all three rows)
   the LM head runs on the last position only, as serving prefill does.
8. **Row 3** (`QuarotNARLlamaMLP`): `down_proj = Sequential(NARDownTransform, Quantizer, Linear4bit)`
   where the last two are the very objects row 2 uses.  Nothing else differs from row 2.

### Driver (`e28_bench.py`)

9. Random weights: `no_init_weights` leaves parameters uninitialised, so fp16
   parameters are drawn N(0, 0.02²), norms set to 1, `Linear4bit.weight_scales`
   (zero by construction) set to U(0.001, 0.003); INT4 codes are QuaRot's own
   `randint`.  `tie_weights()` is called explicitly (skipped under `no_init_weights`)
   so Llama-3.2-3B's tied LM head is counted once.
10. The fp16 baseline row is QuaRot's own `QuarotFP16LlamaForCausalLM` (HF Llama +
    flash-attn prefill + QuaRot's FlashInfer fp16 paged-KV decode).  QuaRot's fp16 KV
    kernels are half-precision only, so the "bf16" row of the spec is run in **fp16**
    (identical tensor-core throughput on the A40).

## The NAR row's kernel (`nar_r4_kernel.py`)

Kernel A is `nar.kernels.r4_fused_v3.rank_projection_dot_kernel` unchanged
(Y' carried as two fp16 terms since QuaRot's pipeline dtype is fp16).  Kernel B
keeps E17 v3's structure but, for the A40, runs the block-Hadamard-128 as one
`tl.dot` with a ±1 fp16 H₁₂₈ (E17's shuffle FWHT is 2.4× the copy floor on
sm_86) and the rank-k correction as three fp16 hi/lo `tl.dot`s, and stores fp16
instead of E17's group-128 INT4 codes because QuaRot's GEMM consumes per-token
symmetric INT4 from QuaRot's own quantizer (kept as the shared stage).  Every
config is verified against the fp32 reference before it may be timed (E17 v3
protocol; no `triton.autotune`); the selection per token count is in
`results/e28/<model>/nar_kernel_selection.json` together with the precision of
the selected pipeline on every layer's factors and, for comparison, the
precision of QuaRot's own fp16 Hadamard.  Kernels are launched through Triton's
pre-bound `CompiledKernel` runner (checked bit-for-bit against the generic
launch) because decode in this pipeline is CPU-launch-bound.
