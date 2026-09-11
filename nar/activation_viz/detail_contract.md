# Local surface revision contract

This user-directed revision supersedes the earlier identical overview/detail,
height-column and closed-sidewall presentation rules. The earlier records are
historical; this is not a retroactive preregistration.

Question: what local activation structure is present before activation QDQ,
and what remains after signed per-token g128 centering? No method is required
to look better. Quantitative grid: columns hold Blocks 1, 13, 24, 36; rows hold
Unrotated reference, Hadamard, PrismQuant (k=8), PrismQuant (k=max). Paired
local rows share canonical inputs. End-to-end uses the matching paired-local
unrotated norm-fused FP32 reference above three observed quantized forwards.

Python/Matplotlib native 3D upper surfaces use sample 0, tokens [0,128), channels
[0,512), all 65,536 vertices and all 64,897 adjacent cells. Heights are linear,
unchanged absolute raw values or absolute residuals computed from the complete
signed tensor before cropping. No pooling, data stride, zero padding, smoothing,
columns or walls. Four complete groups are located by floor-only boundaries
at 127.5, 255.5 and 383.5.

The common orthographic camera is elevation 28, azimuth -55, aspect (2,1.2,1).
Viridis and PowerNorm(gamma=0.5) map colors only; each column shares 1.03 times
the maximum of its displayed methods. All-zero columns explicitly use a
display-only upper bound of 1. The linear-color control uses identical heights,
camera and limits. Rotated-only zoom explicitly uses its own shared limits.

PDF/SVG preserve vector text/axes and 600-dpi rasterized surfaces. A 7-point
target applies at the supplied physical dimensions; do not shrink a dense
four-column matrix below readable text size. Single-block panels support
narrower placement. Every export has geometry, hashes, alignment and PDF audits.
Debug exports mark exact corners and real maxima; PDF pages are rasterized
again for visual inspection. Uncertainty is not drawn because each surface is
one fixed observed sample, not an estimate across independent models.

Overview remains the full-domain context; metrics and exact ECDFs retain all
eight samples. The local window need not contain global outliers. Raw tensors,
frozen factors, quantizers, measured statistics and supplementary failures are
immutable. Integrity snapshots and the publication manifest record this.
