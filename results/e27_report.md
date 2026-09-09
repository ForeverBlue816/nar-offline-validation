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

**llama31_8b: complete.** D triggered: False.

| Site | Variant | Mean PPL | Paired delta vs A [90% CI] |
|---|---|---:|---:|
| qkv_only | A_full | 6.22450 | +0.00000 [+0.00000, +0.00000] |
| qkv_only | B_top1 | 6.22752 | +0.00302 [+0.00080, +0.00524] |
| qkv_only | B_top1_no_bos | 6.22637 | +0.00187 [-0.00189, +0.00562] |
| qkv_only | C_top1pct | 6.22754 | +0.00304 [+0.00171, +0.00436] |
| qkv_only | C_top1pct_no_bos | 6.22393 | -0.00058 [-0.00625, +0.00510] |
| qkv_only | hadamard | 6.24293 | +0.01843 [+0.00692, +0.02994] |
| both | A_full | 6.28493 | +0.00000 [+0.00000, +0.00000] |
| both | B_top1 | 6.30094 | +0.01602 [+0.01175, +0.02028] |
| both | B_top1_no_bos | 6.34178 | +0.05686 [+0.04863, +0.06509] |
| both | C_top1pct | 6.29289 | +0.00796 [+0.00177, +0.01414] |
| both | C_top1pct_no_bos | 6.32933 | +0.04440 [+0.03652, +0.05228] |
| both | hadamard | 6.34626 | +0.06133 [+0.05641, +0.06626] |
| down_only | A_full | 6.26217 | +0.00000 [+0.00000, +0.00000] |
| down_only | B_top1 | 6.27918 | +0.01701 [+0.01288, +0.02113] |
| down_only | B_top1_no_bos | 6.31585 | +0.05368 [+0.04803, +0.05934] |
| down_only | C_top1pct | 6.27127 | +0.00910 [+0.00261, +0.01560] |
| down_only | C_top1pct_no_bos | 6.30564 | +0.04347 [+0.03716, +0.04978] |
| down_only | hadamard | 6.31542 | +0.05325 [+0.04954, +0.05695] |

| Site | Variant | Mean f | Mean range/Hadamard | Mean NMSE | Mean top-8 angle vs A (deg) |
|---|---|---:|---:|---:|---:|
| qkv | A_full | 0.43575 | 0.73485 | 0.0054938 | 1.2996e-06 |
| qkv | B_top1 | 0.27043 | 0.82986 | 0.006936 | 55.951 |
| qkv | B_top1_no_bos | 0.27226 | 0.82934 | 0.0069304 | 55.763 |
| qkv | C_top1pct | 0.34507 | 0.79044 | 0.0062969 | 47.989 |
| qkv | C_top1pct_no_bos | 0.34664 | 0.78991 | 0.0062901 | 47.918 |
| qkv | hadamard | 0.0064948 | 1 | 0.00988 | nan |
| down | A_full | 0.32433 | 0.74597 | 0.0058055 | 1.8022e-06 |
| down | B_top1 | 0.15167 | 0.83907 | 0.0071962 | 77.029 |
| down | B_top1_no_bos | 0.10775 | 0.85695 | 0.0072389 | 76.467 |
| down | C_top1pct | 0.21493 | 0.81063 | 0.0067322 | 66.373 |
| down | C_top1pct_no_bos | 0.16703 | 0.8323 | 0.0068155 | 66.964 |
| down | hadamard | 0.0071581 | 1 | 0.0093655 | nan |

B_top1, both sites: A lower than B with a paired interval above zero.

B_top1_no_bos, both sites: A lower than B with a paired interval above zero.

5 B layer/site estimates have fewer than eight identified directions. Their required top-8 angle rows include the preregistered null-space completion and must not be interpreted as eight data-identified directions.

Layer/seed rows and all eight angles are retained in the CSV. Repeated seeds are not independent layer replicates.

<!-- E27_RESULTS_END -->
