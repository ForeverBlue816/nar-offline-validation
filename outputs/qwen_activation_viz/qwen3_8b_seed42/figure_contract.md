# Predeclared figure contract

Question: how do the frozen deployed rotations redistribute activation energy and change actual group-128 quantization error on matched Qwen3-8B-Base inputs? No direction of improvement is assumed.

Quantitative grid: A shows post-rotation/pre-QDQ magnitude; B shows the magnitude after signed group centering (analysis only); group-range ECDF and reconstruction metrics distinguish energy concentration from quantization quality. q_proj is a control using global R1. End-to-end results are separate because their canonical activations differ.

Python/Matplotlib; Times New Roman; white background, blue-to-orange amplitude scale; elevation 25, azimuth -60. Four columns: zero-based layers [0,12,23,35]. Four rows: identity, Hadamard, frozen k=8, frozen k=max. Sample 0 is the primary surface; all eight samples enter full-resolution metrics. Select 8 of the 146 existing WikiText-2 test windows using torch.randperm with seed 42, retain order. Sequence length 2048, batch size 1, no padding or generation. Rotation seed 0. No fitting on test data.

Overview: token 0 alone plus 127 contiguous bins covering tokens 1..2047, 256 contiguous channel bins, max-absolute pooling only. Detail: tokens [0,128), channels [0,512), four complete quantizer groups, no pooling. Same mapping for all methods. Residual is computed before magnitude or pooling. Same z/color range per layer/site/quantity/view across methods; different columns may use different ranges. Rotated channel coordinates denote a new basis. Keep all peaks, token 0 and linear zero-based scales.

Formats: 600 dpi PNG, PDF and editable-text SVG; surfaces may be rasterized. Full signed FP32 tensors remain in the experiment artifact volume; Git holds display caches, complete full-precision tables, hashes, provenance, source and figures. Export individual panels, method rows and 4x4 matrices. Final matrices approximately 183 mm square, glyph floor 5 pt; panels also exported at larger readable sizes.

Predeclared checks: finite floating data; exact shared token IDs and shared canonical tensor; per-token squared-energy relative error <=2e-5 for FP32 rotations; inverse round trip <=1e-5; compensated small linear relative error <=2e-5; norm-fusion logit relative error <=1e-5; capture enabled/disabled logits bitwise equal; group range residual discrepancy <=2e-6 times max(1,range); rho+f discrepancy <=2e-6. FP64 energy accumulation; zero-energy ratios null. No tolerances relaxed after observation. Actual fake-quantizer fp16 scale/offset semantics are reused, including its underflow guard.
