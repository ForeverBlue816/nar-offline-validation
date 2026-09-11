# Capture-site audit

Model: Qwen/Qwen3-8B-Base, revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4. The runtime checks the real architecture: 36 decoder blocks, hidden width 4096, MLP width 12288, 32 query heads, 8 KV heads, head dimension 128. Full forward source is preserved in the run manifest. `q_norm` and `k_norm` remain per-head affine RMSNorms before RoPE; they are not fused into Q/K projections.

Full module paths use `model.layers.{0,12,23,35}.mlp.down_proj` and `model.layers.{0,12,23,35}.self_attn.q_proj`.

## Paired local

The original BF16 checkpoint is loaded into FP32 containers. Input/post-attention RMSNorm affine weights are fused into their consumers, and final norm into lm_head, using exactly the norm-affine portion of E14. No rotation or quantization is active in this reference forward. At down_proj, X is the real SiLU(gate) times up product. At q_proj, X is the norm-fused residual input. Every method receives the same cloned logical tensor, with identical hashes logged.

For q_proj, the transformation is the deployed global R1 (layer argument 0), never a per-layer fitted substitute. For down_proj, use that block's frozen E18v2 R4. The repository row-vector functions determine the transformation orientation. NAR uses Householder/WY, permutation, diagonal signs and group Hadamard; the Hadamard baseline uses the repository full-width structured Hadamard. No D/P factor is pre-folded into the unrotated reference gate/up path.

## End to end

Load the existing E22 GPTQ g128_asym checkpoint with `e14.load_quantized_model`, which norm-fuses and rotates base weights before replacing all seven linear weights per block with saved QDQ states. The actual saved tensor dtype is inspected rather than inferred from an old DONE.json description. These are floating fake-quantized weights, not a packed inference kernel.

`Observer` subclasses the existing E14 RuntimeHooks. Its read-only callback runs at entry to the original `quantize_input` method and calls that method unchanged. E14 registers down rotation before activation quantization, so the callback sees the complete online R4 result once. No extra rotation is applied to observed end-to-end tensors. At q_proj, R1 is already represented by the globally rotated residual stream.

Runtime activation QDQ operates at q/k/v/o/gate/up/down inputs. Nominal W/A/K/V widths are 4/4/4/4; embeddings, head, normalization and the recent KV residual remain floating. Activations use per-token contiguous channel groups of 128, min/max clipping, FP16 scale and FP16 real offset. Key QDQ uses token-axis groups of 32; value QDQ uses channel groups of 128. The latest 32-token residual is handled by E14's causal correction. Custom attention invokes K/V QDQ during prefill even with `use_cache=False`; actual calls are counted during the eight recorded forwards, excluding the capture-disabled control.

## Integrity controls

Batch size is one. All 2048 positions are retained, including the actual prefix token inserted by the established preprocessing; no padding occurs. The manifest distinguishes tokenizer BOS metadata from the observed token ID. `use_cache=False` disables cache persistence, not the custom KV quantizers. No autoregressive generation is performed.

An observation of the final norm output records every token's final hidden state. Enabled/disabled capture checks require bitwise equality of all these states and last-token logits. All-position logits are determined by these identical final states and unchanged lm_head; the large full-vocabulary head output is only materialized for the last token. Norm-affine equivalence uses a 128-token probe; rotation/compensated-linear/WY checks use real captured values. All callbacks are removed after use. The E14 Python path is the actual existing evaluation implementation; no fused-kernel equivalence is assumed.

All plotted values are floating Y immediately before QDQ. Statistics compute QDQ(Y) using the actual runtime function; centered residual E is an analysis view and is never substituted into the NMSE quantizer.
