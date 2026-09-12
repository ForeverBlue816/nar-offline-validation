# Quantizer contract: three distinct paths

The source of truth is the pinned code and binaries in `source_manifest.json`, not a shared W4A4 label. A 128-channel Hadamard computation block is not a quantization group.

| Property | E28-v2 integer kernel swap | E17 native R4 | Paper accuracy path (E14/E22 g128_asym) |
| --- | --- | --- | --- |
| W | Signed INT4, one FP16 scale/output channel; benchmark packed bytes0..255 cover both signed nibbles; legacy1..6 initialization is retained as a failed predecessor | No complete W GEMM in this local benchmark | GPTQ W4, group128 along input/reduction axis, per-output/group scales and asymmetric integer zero point |
| A | Signed INT4, symmetric, one scale per complete token row | Unsigned INT4, asymmetric, per token/channel-group128 | Asymmetric A4, per token/channel-group128; fake quantization in floating model |
| A scale/offset | FP16 `max(abs(x))/7`, clip ratio1; no offset and no zero-scale guard | FP16 `(max-min)/15`; if range0, scale1; FP16 real offset=min | `dynamic_asym_int4`: FP16 group scale and real-valued min offset; follow recorded accuracy protocol |
| Rounding/packing | FP16 division, CUDA nearest-even, saturation[-8,7], even channel in low nibble, odd in high | FP32 quotient using rounded FP16 scale/offset; floor(q+0.5), clamp[0,15], low/even nibble | Fake quant/dequant; not evidence of a compatible packed GEMM |
| Accumulator/output | CUTLASS signed INT4→INT32; existing dequant casts INT32 to FP16 before multiplying FP16 row/column scales; FP16 output | A partials FP32; B pack codes UINT8 and scales/offset FP16 | Floating GEMM on fake-quantized weights/activations; not measured integer deployment |
| K | Per token/head, affine over head channels128; rotate K with normalized H128 before packing | Not included | K4 groups32 along token axis per channel after RoPE |
| V | Per token/head, affine over head channels128 | Not included | V4 groups along head channels128 per token |
| KV affine convention | scale=clamp(max-min,min1e-5)/15; zero=-min; reconstruction=q*scale-zero, nearest-even | Not included | Per-group real offset/scales from actual accuracy hooks |
| KV residual | None; all stored K/V quantized immediately | Not included | Residual window32; K prefix updated in blocks32 and causal V residual policy |
| Prefill attention | FlashAttention reads original FP16 K/V while writing quantized cache; subsequent decode reads packed cache | Not included | Accuracy hooks simulate causal quantized history; distinct policy |
| Unquantized parts | FP16 embedding/head, norm, RoPE, residual and nonlinearities; FP16 reference has real FP16 cache | Transform reference only | Floating embedding/head/norm/nonlinearities, fake-quantized linear model |

E28 sources: `external/quarot/quarot/nn/{linear,quantization}.py`, `kernels/{quant.cu,bindings.cpp}`, `transformers/kv_cache.py`, plus `quarot-llama3/modeling_llama3.py`. E17 source: `nar/kernels/r4_fused_v2.py::_quantize_pack_store`, reused by v3. Accuracy sources: `nar/e14_w4a4kv4.py::WEIGHT_PROTOCOLS`, `K_TOKEN_GROUP=32`, `KV_RESIDUAL_LENGTH=32`, `RuntimeHooks`, `nar/quarot_gptq.py`, and `nar/e22_qwen3_family.py::ACTIVATION_KIND`. E22 is read only as a format source; no Qwen workload was added.

E28 random weights have **no claim of valid offline R1/R2/R4 compensation**. Online head mixing (R2) and query/key H128 in cached attention (R3) are common to both integer rows. The down-input online R4 is full-width QuaRot Hadamard versus folded NAR block-Hadamard/low-rank correction. The exported NAR factors assume a signed permutation already absorbed into gate/up weights. Arbitrary identical random tensors are a performance control, not a legitimately compensated real model. Paper accuracy checkpoints instead carry method-specific offline compensation; those related real weights should not be forced to have identical bytes.

## Why the current GEMM cannot consume the paper format

The current interface computes one integer dot product and multiplies a scale for its activation row and weight output channel. For groups g along the reduction dimension, the required result instead contains a sum of individually scaled partial products, plus offset corrections:

`sum_g [sA_g*sW_g*sum_j qA_j*qW_j + sA_g*bW_g*sum_j qA_j + bA_g*sW_g*sum_j qW_j + |g|*bA_g*bW_g]`.

The current single INT32 accumulation and final row/column scale do not expose those group partials or offset terms. A different KV residual/grouping implementation is also needed. Dropping offsets, changing to per-token symmetric quantization, or dequantizing into FP16 GEMM would change the contract and cannot establish paper-native integer deployment.

`checkpoint_audit.json` records the local files at freeze time: no complete matching k8 manifest exists for these two models. Existing partial layer dumps and the fake-quantized E14 loader do not satisfy the integer loader requirement. This is BLOCKED, and no full PPL or model-quality claim is made.
