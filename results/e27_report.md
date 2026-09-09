# E27 — which tokens define the subspace

<!-- E27_RESULTS_BEGIN -->

### E27 execution and results

**llama32_3b: complete.** D triggered: False.

| Site | Variant | Mean PPL | Paired delta vs A [90% CI] |
|---|---|---:|---:|
| qkv_only | A_full | 7.64166 | +0.00000 [+0.00000, +0.00000] |
| qkv_only | B_top1 | 7.64382 | +0.00216 [-0.00689, +0.01120] |
| qkv_only | B_top1_no_bos | 7.64595 | +0.00428 [-0.00001, +0.00858] |
| qkv_only | C_top1pct | 7.64339 | +0.00173 [-0.00858, +0.01203] |
| qkv_only | C_top1pct_no_bos | 7.64346 | +0.00179 [-0.00040, +0.00399] |
| qkv_only | hadamard | 7.65860 | +0.01694 [+0.01056, +0.02332] |
| both | A_full | 7.70528 | +0.00000 [+0.00000, +0.00000] |
| both | B_top1 | 7.72271 | +0.01743 [-0.00156, +0.03641] |
| both | B_top1_no_bos | 7.80025 | +0.09497 [+0.09182, +0.09812] |
| both | C_top1pct | 7.71123 | +0.00595 [-0.01174, +0.02364] |
| both | C_top1pct_no_bos | 7.80603 | +0.10075 [+0.07228, +0.12923] |
| both | hadamard | 7.77113 | +0.06585 [+0.03907, +0.09263] |
| down_only | A_full | 7.67539 | +0.00000 [+0.00000, +0.00000] |
| down_only | B_top1 | 7.68765 | +0.01226 [+0.00301, +0.02151] |
| down_only | B_top1_no_bos | 7.76575 | +0.09036 [+0.07366, +0.10707] |
| down_only | C_top1pct | 7.68838 | +0.01299 [+0.00693, +0.01905] |
| down_only | C_top1pct_no_bos | 7.77222 | +0.09683 [+0.08343, +0.11022] |
| down_only | hadamard | 7.71257 | +0.03719 [+0.03314, +0.04123] |

| Site | Variant | Mean f | Mean range/Hadamard | Mean NMSE | Mean top-8 angle vs A (deg) |
|---|---|---:|---:|---:|---:|
| qkv | A_full | 0.39417 | 0.76243 | 0.0059072 | 1.3931e-06 |
| qkv | B_top1 | 0.2151 | 0.86294 | 0.0074857 | 59.752 |
| qkv | B_top1_no_bos | 0.21726 | 0.8622 | 0.0074728 | 59.283 |
| qkv | C_top1pct | 0.30064 | 0.82007 | 0.0067668 | 50.524 |
| qkv | C_top1pct_no_bos | 0.30042 | 0.82016 | 0.0067668 | 50.507 |
| qkv | hadamard | 0.0050043 | 1 | 0.0099183 | nan |
| down | A_full | 0.33278 | 0.74691 | 0.0057303 | 1.6141e-06 |
| down | B_top1 | 0.15391 | 0.84503 | 0.0071797 | 77.004 |
| down | B_top1_no_bos | 0.10332 | 0.86097 | 0.0071852 | 76.146 |
| down | C_top1pct | 0.22587 | 0.81186 | 0.006647 | 66.187 |
| down | C_top1pct_no_bos | 0.16405 | 0.84128 | 0.0067922 | 67.032 |
| down | hadamard | 0.0073489 | 1 | 0.0092203 | nan |

B_top1, both sites: paired interval overlaps zero; inconclusive ordering.

B_top1_no_bos, both sites: A lower than B with a paired interval above zero.

7 B layer/site estimates have fewer than eight identified directions. Their required top-8 angle rows include the preregistered null-space completion and must not be interpreted as eight data-identified directions.

Layer/seed rows and all eight angles are retained in the CSV. Repeated seeds are not independent layer replicates.

**llama31_8b:** pending completion; no conclusion yet.

<!-- E27_RESULTS_END -->
