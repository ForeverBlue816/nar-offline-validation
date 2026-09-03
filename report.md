# NAR offline tensor validation — corrected gate and activation extension

## Outcome

The original b=128 K phenomenon gate was dimensionally mis-specified: head_dim=128 gives only one b=128 group and therefore one DC slot. It is retained as an archival measurement, not used for the E1 decision. The valid pre-registered K gates are b=32 and b=64; no E1 or E2 row was rerun.

- **Corrected E1 K criterion: PASS**. b=32: range reduction 20.489%, attribution 26.954%, held-out retention 99.900%; b=64: range reduction 12.612%, attribution 20.667%, held-out retention 99.307%.
- **E2 criterion over every frozen NAR row: PASS**. Every E2 result remains valid and is listed below.
- **Corrected method-promising criterion: PASS**.
- **E0 implementation sanity: PASS**.

## Frozen protocol and comparison rules

All comparisons are paired on identical tensors. Dynamic asymmetric INT4 uses one fp16 scale and one fp16 real-valued offset for each group and NMSE is `sum((x_hat-x)^2)/sum(x^2)`. No weight quantization, GPTQ, configuration search, or end-to-end W4A4 run was performed. E1b reads the prior K dump; E1c/E1d use one new forward capture of the same 128 WikiText-2 train sequences. E2 is read only.

## Corrected E1 — frozen K results

| b | method | mean range | mean NMSE |
|---|---|---|---|
| 32 | bf16 | 8.93598 | 0 |
| 32 | hadamard | 7.72264 | 0.00587887 |
| 32 | identity | 8.93598 | 0.0091897 |
| 32 | nar | 6.14034 | 0.00377166 |
| 32 | nar_rope | 6.94722 | 0.00485515 |
| 64 | bf16 | 12.1472 | 0 |
| 64 | hadamard | 8.6538 | 0.00757424 |
| 64 | identity | 12.1472 | 0.0159797 |
| 64 | nar | 7.56239 | 0.00583484 |
| 64 | nar_rope | 8.15062 | 0.00678913 |

| b | range reduction | top-k attribution | held-out retention | pass |
|---|---|---|---|---|
| 32 | 0.204892 | 0.269542 | 0.998999 | True |
| 64 | 0.126119 | 0.206672 | 0.993069 | True |

### NAR-RoPE is dominated and dropped

Plain NAR has lower range and lower NMSE than NAR-RoPE in every one of 28 layers at both valid group sizes; it also has lower PPL at every paired E2 seed where NAR-RoPE exists. NAR-RoPE is therefore strictly dominated in the observed comparisons and no further NAR-RoPE work was run.

| b | plain NAR lower-range layers | plain NAR lower-NMSE layers | layers |
|---|---|---|---|
| 32 | 28 | 28 | 28 |
| 64 | 28 | 28 | 28 |

## E1b — plain-NAR position check

The exact stored pre-RoPE K samples were re-rotated using model RoPE at positions 0/512/1024/2048. This is offline tensor analysis, not a model rerun.

| b | position | method | mean range | mean NMSE | reduction vs Had |
|---|---|---|---|---|---|
| 32 | 0 | bf16 | 9.00604 | 0 | -0.164264 |
| 32 | 0 | hadamard | 7.73433 | 0.00589957 | 0 |
| 32 | 0 | identity | 9.00604 | 0.00929213 | -0.164264 |
| 32 | 0 | nar | 6.22136 | 0.00386866 | 0.196025 |
| 32 | 512 | bf16 | 8.931 | 0 | -0.154981 |
| 32 | 512 | hadamard | 7.73475 | 0.00589262 | 0 |
| 32 | 512 | identity | 8.931 | 0.00918622 | -0.154981 |
| 32 | 512 | nar | 6.14909 | 0.00378298 | 0.205271 |
| 32 | 1024 | bf16 | 8.96045 | 0 | -0.157007 |
| 32 | 1024 | hadamard | 7.74547 | 0.00591224 | 0 |
| 32 | 1024 | identity | 8.96045 | 0.00922734 | -0.157007 |
| 32 | 1024 | nar | 6.12395 | 0.00375294 | 0.20978 |
| 32 | 2048 | bf16 | 8.94702 | 0 | -0.159807 |
| 32 | 2048 | hadamard | 7.71534 | 0.00586399 | 0 |
| 32 | 2048 | identity | 8.94702 | 0.00920616 | -0.159807 |
| 32 | 2048 | nar | 6.20574 | 0.00384567 | 0.196083 |
| 64 | 0 | bf16 | 12.2065 | 0 | -0.411703 |
| 64 | 0 | hadamard | 8.64831 | 0.0075586 | 0 |
| 64 | 0 | identity | 12.2065 | 0.016105 | -0.411703 |
| 64 | 0 | nar | 7.61476 | 0.00590891 | 0.119838 |
| 64 | 512 | bf16 | 12.1381 | 0 | -0.404233 |
| 64 | 512 | hadamard | 8.64453 | 0.0075604 | 0 |
| 64 | 512 | identity | 12.1381 | 0.0159692 | -0.404233 |
| 64 | 512 | nar | 7.57466 | 0.00585343 | 0.124286 |
| 64 | 1024 | bf16 | 12.162 | 0 | -0.40462 |
| 64 | 1024 | hadamard | 8.65966 | 0.00758353 | 0 |
| 64 | 1024 | identity | 12.162 | 0.0160262 | -0.40462 |
| 64 | 1024 | nar | 7.55689 | 0.0058241 | 0.127882 |
| 64 | 2048 | bf16 | 12.1451 | 0 | -0.401623 |
| 64 | 2048 | hadamard | 8.66659 | 0.00758866 | 0 |
| 64 | 2048 | identity | 12.1451 | 0.0159722 | -0.401623 |
| 64 | 2048 | nar | 7.6056 | 0.00588869 | 0.12267 |

## E1c — wide activation inputs, b=128

The dump retains every token as exact bf16 bit patterns at q_proj input (post-input-RMSNorm residual, n=3072, 24 DC slots) and down_proj input (post-SiLU gated MLP product, n=8192, 64 DC slots). Top directions use all 262144 tokens with a fixed randomized symmetric eigensolver: oversampling 16, one power iteration, three full passes, and published Ritz residuals. Range/NMSE and the greedy residual-energy permutation use positions `0,32,...,2016` from all 128 sequences (8192 paired token vectors per layer). Hadamard is a random-sign full-feature transform (H8192 or fixed Paley H12 tensor H256 for n=3072). Each NAR k uses low-energy fillers for unused DC slots and the same H128.

| site | method | layers | mean_group_range | mean_relative_quantization_error_nmse | mean_range_reduction_vs_hadamard | mean_nmse_delta_vs_hadamard |
|---|---|---|---|---|---|---|
| q_input | bf16 | 28 | 3.29754 | 0 | -0.531806 | -0.00991938 |
| q_input | identity | 28 | 3.29754 | 0.0265487 | -0.531806 | 0.0166294 |
| q_input | hadamard_full | 28 | 2.17381 | 0.00991938 | 0 | 0 |
| q_input | nar_kmax | 28 | 1.67804 | 0.00590774 | 0.23755 | -0.00401163 |
| down_input | bf16 | 28 | 0.644721 | 0 | -0.532605 | -0.00921417 |
| down_input | identity | 28 | 0.644721 | 0.0301795 | -0.532605 | 0.0209653 |
| down_input | hadamard_full | 28 | 0.407325 | 0.00921417 | 0 | 0 |
| down_input | nar_kmax | 28 | 0.282991 | 0.0057301 | 0.253007 | -0.00348407 |

Range reduction is shown both as a ratio of layer-mean ranges and as the mean of paired per-layer reductions:

| site | ratio of mean ranges | mean paired-layer reduction |
|---|---|---|
| q_input | 0.228065 | 0.23755 |
| down_input | 0.305247 | 0.253007 |

![E1c range versus k](results/llama32_3b/e1c_range_vs_k.png)

![E1c energy fit](results/llama32_3b/e1c_energy_fit.png)

The fit is pooled OLS over paired layer-k rows, `range(k)/range(0) = intercept + slope*sqrt(1-f)`:

| site | fit | intercept | slope | r_squared | rmse | points |
|---|---|---|---|---|---|---|
| q_input | OLS normalized_range = intercept + slope * sqrt(1-f), pooled over paired layer-k rows | 0.0908306 | 0.917523 | 0.966019 | 0.0102667 | 700 |
| down_input | OLS normalized_range = intercept + slope * sqrt(1-f), pooled over paired layer-k rows | 0.536016 | 0.403782 | 0.716882 | 0.0402828 | 1820 |

Randomized eigenspace approximation quality (relative Ritz residual; every direction is in the exact CSV):

| site | median | max |
|---|---|---|
| down_input | 0.173612 | 0.338376 |
| q_input | 0.0942059 | 0.261449 |

## E1d — KIVI-style per-channel K baseline

At the same 4-bit width and b=32/64 metadata grouping, KIVI-style K quantization groups contiguous tokens independently per sequence/head/channel. Per-token methods group channels. All operate on the same full post-RoPE K dump; dominance is decided by global NMSE, with per-layer counts also reported.

| b | bits | method | axis | layers | mean_group_range | global_relative_quantization_error_nmse |
|---|---|---|---|---|---|---|
| 32 | 4 | bf16 | channels | 28 | 9.04655 | 0 |
| 32 | 4 | identity_per_token | channels | 28 | 9.04655 | 0.00920431 |
| 32 | 4 | hadamard_per_token | channels | 28 | 7.81505 | 0.00588228 |
| 32 | 4 | nar_per_token | channels | 28 | 6.21258 | 0.00377893 |
| 32 | 4 | kivi_per_channel | tokens | 28 | 4.01217 | 0.00174985 |
| 64 | 4 | bf16 | channels | 28 | 12.2864 | 0 |
| 64 | 4 | identity_per_token | channels | 28 | 12.2864 | 0.0160151 |
| 64 | 4 | hadamard_per_token | channels | 28 | 8.75643 | 0.00757454 |
| 64 | 4 | nar_per_token | channels | 28 | 7.65093 | 0.0058459 |
| 64 | 4 | kivi_per_channel | tokens | 28 | 4.71912 | 0.00245616 |

| b | kivi_beats_every_per_token_method_global_nmse | layers_where_kivi_beats_every_per_token_method_nmse | layers |
|---|---|---|---|
| 32 | True | 28 | 28 |
| 64 | True | 28 | 28 |

**Per-channel K is the clear winner in this test.** b=32: KIVI 0.00174985 vs best rotation NAR 0.00377893; b=64: KIVI 0.00245616 vs best rotation NAR 0.00584590. It beats every per-token method in all 28/28 layers at both group sizes. For K under these conditions, the standard per-channel axis is preferable to every tested per-token rotation method.

## E2 — frozen KV-only perplexity results

| model | b | method | seeds | mean_ppl | seed_ppl_std | paired_ppl_delta_vs_hadamard | paired_90ci_low | paired_90ci_high | seed_ppls |
|---|---|---|---|---|---|---|---|---|---|
| llama32_3b | 64 | bf16 | 3 | 7.61655 | 0 | -0.119123 | -0.12998 | -0.108267 | 7.61655149;7.61655149;7.61655149 |
| llama32_3b | 64 | hadamard | 3 | 7.73567 | 0.00643994 | 0 | 0 | 0 | 7.74296257;7.733312;7.73075024 |
| llama32_3b | 64 | identity | 3 | 7.80385 | 0 | 0.0681739 | 0.0573171 | 0.0790307 | 7.80384882;7.80384882;7.80384882 |
| llama32_3b | 64 | nar | 3 | 7.70657 | 0.00088966 | -0.029102 | -0.0392892 | -0.0189148 | 7.70725479;7.70556656;7.70689737 |
| llama32_3b | 64 | nar_rope | 3 | 7.72 | 0.00600981 | -0.0156717 | -0.0360164 | 0.00467304 | 7.71336694;7.72507847;7.72156437 |
| llama32_3b | 128 | bf16 | 3 | 7.61655 | 0 | -0.165856 | -0.175301 | -0.15641 | 7.61655149;7.61655149;7.61655149 |
| llama32_3b | 128 | hadamard | 3 | 7.78241 | 0.00560303 | 0 | 0 | 0 | 7.78166119;7.78834558;7.77721422 |
| llama32_3b | 128 | identity | 3 | 8.00706 | 0 | 0.224652 | 0.215206 | 0.234098 | 8.00705928;8.00705928;8.00705928 |
| llama32_3b | 128 | nar | 3 | 7.76527 | 0.00109193 | -0.0171341 | -0.0278942 | -0.00637397 | 7.76443563;7.76487511;7.76650796 |
| llama32_1b | 32 | bf16 | 3 | 9.52734 | 0 | -0.374289 | -0.393114 | -0.355464 | 9.52734406;9.52734406;9.52734406 |
| llama32_1b | 32 | hadamard | 3 | 9.90163 | 0.0111664 | 0 | 0 | 0 | 9.90776974;9.90838613;9.8887445 |
| llama32_1b | 32 | identity | 3 | 10.0253 | 0 | 0.123619 | 0.104794 | 0.142444 | 10.0252528;10.0252528;10.0252528 |
| llama32_1b | 32 | nar | 3 | 9.78784 | 0.00647878 | -0.113791 | -0.123403 | -0.104178 | 9.7886786;9.79386273;9.78098635 |
| llama32_1b | 32 | nar_rope | 3 | 9.83579 | 0.00640406 | -0.0658458 | -0.0945095 | -0.0371821 | 9.83604074;9.82926079;9.8420614 |

## Negative findings, caveats, and what remains unsure

- NAR-RoPE is dominated by plain NAR in every available paired check and is no longer pursued.
- KIVI-style per-channel K beats every tested per-token rotation by global NMSE and in every layer at both b=32/64.
- The E1c eigenspaces are deterministic randomized approximations, not dense 8192x8192 decompositions; Ritz residuals are published per direction and are non-negligible for tail directions.
- E1c stores all tokens and uses all of them for the top-direction solve, but evaluates range/NMSE and balances Pi on a fixed position stride to bound repeated k-sweep cost.
- Per-channel and per-token range means describe different axes, so E1d fairness is decided by paired NMSE; metadata count and bit width are matched.
- E1c dumps are deliberately retained under project storage for the separately scoped E3 FP4 E2M1 and E4 two-level NVFP4 checks.

## Go / no-go

**GO to the already-scoped activation-shape checks, without tuning.** This decision uses b=32/64 K E1 and every frozen E2 row; the newly measured E1c/E1d results are mechanistic follow-ups and are reported regardless of sign.

## Reproduction artifacts

Exact tables and plots are under `results/`; commands and logs are under `runs/`. Large raw bf16 dumps and randomized eigenspace checkpoints remain under `activations/` in project storage and are excluded from Git.

# Activation continuation — E5–E8

## Scope decision

K is closed: KIVI-style per-channel quantization remains the clear K result, consistent with the zero-point null-space interpretation because channel-persistent outliers are constant along the token grouping axis. No additional K experiment was run. E5–E8 concern activation sites where per-token quantization is forced, plus the standard per-token V cache.

## E5 — activation-only perplexity proxy

Only post-RMSNorm q/k/v inputs and/or down_proj inputs are dynamically asymmetric group-128 INT4 fake-quantized. Scale and real-valued offset are fp16. Weights remain bf16; each activation rotation is folded algebraically into the corresponding q/k/v or down_proj weight rows. KV and all other activations remain bf16. Results use the same 64 WikiText-2 test chunks, three paired rotation seeds, and two-sided paired 90% Student-t intervals over seed-level PPL differences. The 8B model was included because a GPU was available.

| model | site | method | mean_ppl | ppl_delta_vs_bf16 | paired_ppl_delta_vs_hadamard | paired_90ci_low_vs_hadamard | paired_90ci_high_vs_hadamard | paired_ppl_delta_vs_identity | paired_90ci_low_vs_identity | paired_90ci_high_vs_identity |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | none | bf16 | 7.61682 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| Llama-3.2-3B | q/k/v only | identity | 7.77782 | 0.160991 | 0.119214 | 0.113259 | 0.12517 | 0 | 0 | 0 |
| Llama-3.2-3B | q/k/v only | hadamard | 7.6586 | 0.0417771 | 0 | 0 | 0 | -0.119214 | -0.12517 | -0.113259 |
| Llama-3.2-3B | q/k/v only | nar | 7.64166 | 0.0248368 | -0.0169403 | -0.02332 | -0.0105605 | -0.136155 | -0.144086 | -0.128223 |
| Llama-3.2-3B | both sites | identity | 8.14979 | 0.532965 | 0.37866 | 0.368303 | 0.389017 | 0 | 0 | 0 |
| Llama-3.2-3B | both sites | hadamard | 7.77113 | 0.154305 | 0 | 0 | 0 | -0.37866 | -0.389017 | -0.368303 |
| Llama-3.2-3B | both sites | nar | 7.70528 | 0.0884571 | -0.0658479 | -0.0926304 | -0.0390653 | -0.444508 | -0.461528 | -0.427488 |
| Llama-3.2-3B | down_proj only | identity | 7.94256 | 0.32574 | 0.229991 | 0.225224 | 0.234758 | 0 | 0 | 0 |
| Llama-3.2-3B | down_proj only | hadamard | 7.71257 | 0.0957489 | 0 | 0 | 0 | -0.229991 | -0.234758 | -0.225224 |
| Llama-3.2-3B | down_proj only | nar | 7.67539 | 0.0585638 | -0.037185 | -0.0412283 | -0.0331418 | -0.267176 | -0.269692 | -0.264661 |
| Llama-3.2-1B | none | bf16 | 9.52761 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| Llama-3.2-1B | q/k/v only | identity | 9.94736 | 0.419758 | 0.327455 | 0.320098 | 0.334812 | 0 | 0 | 0 |
| Llama-3.2-1B | q/k/v only | hadamard | 9.61991 | 0.0923027 | 0 | 0 | 0 | -0.327455 | -0.334812 | -0.320098 |
| Llama-3.2-1B | q/k/v only | nar | 9.57707 | 0.04946 | -0.0428427 | -0.0533558 | -0.0323296 | -0.370298 | -0.383379 | -0.357217 |
| Llama-3.2-1B | both sites | identity | 10.7611 | 1.23352 | 0.823906 | 0.797587 | 0.850225 | 0 | 0 | 0 |
| Llama-3.2-1B | both sites | hadamard | 9.93722 | 0.409615 | 0 | 0 | 0 | -0.823906 | -0.850225 | -0.797587 |
| Llama-3.2-1B | both sites | nar | 9.71078 | 0.183174 | -0.226441 | -0.256454 | -0.196429 | -1.05035 | -1.05695 | -1.04374 |
| Llama-3.2-1B | down_proj only | identity | 10.2945 | 0.766851 | 0.483249 | 0.463142 | 0.503355 | 0 | 0 | 0 |
| Llama-3.2-1B | down_proj only | hadamard | 9.81121 | 0.283603 | 0 | 0 | 0 | -0.483249 | -0.503355 | -0.463142 |
| Llama-3.2-1B | down_proj only | nar | 9.65874 | 0.131138 | -0.152464 | -0.175749 | -0.12918 | -0.635713 | -0.638895 | -0.63253 |
| Llama-3.1-8B | none | bf16 | 6.20401 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| Llama-3.1-8B | q/k/v only | identity | 6.43919 | 0.235182 | 0.196255 | 0.182628 | 0.209882 | 0 | 0 | 0 |
| Llama-3.1-8B | q/k/v only | hadamard | 6.24293 | 0.0389269 | 0 | 0 | 0 | -0.196255 | -0.209882 | -0.182628 |
| Llama-3.1-8B | q/k/v only | nar | 6.2245 | 0.0204964 | -0.0184306 | -0.0299405 | -0.00692064 | -0.214685 | -0.216913 | -0.212458 |
| Llama-3.1-8B | both sites | identity | 6.82967 | 0.625668 | 0.483414 | 0.4814 | 0.485429 | 0 | 0 | 0 |
| Llama-3.1-8B | both sites | hadamard | 6.34626 | 0.142254 | 0 | 0 | 0 | -0.483414 | -0.485429 | -0.4814 |
| Llama-3.1-8B | both sites | nar | 6.28493 | 0.0809206 | -0.0613332 | -0.0662591 | -0.0564072 | -0.544747 | -0.548787 | -0.540708 |
| Llama-3.1-8B | down_proj only | identity | 6.57405 | 0.37004 | 0.258632 | 0.254754 | 0.262509 | 0 | 0 | 0 |
| Llama-3.1-8B | down_proj only | hadamard | 6.31542 | 0.111408 | 0 | 0 | 0 | -0.258632 | -0.262509 | -0.254754 |
| Llama-3.1-8B | down_proj only | nar | 6.26217 | 0.0581632 | -0.0532453 | -0.0569542 | -0.0495364 | -0.311877 | -0.312873 | -0.310881 |

![E5 PPL deltas](results/activation/e5_ppl_delta.png)

Paired NAR results:

- Llama-3.2-3B q/k/v only: NAR-Hadamard -0.016940 (90% CI [-0.023320, -0.010561]), NAR-identity -0.136155
- Llama-3.2-3B both sites: NAR-Hadamard -0.065848 (90% CI [-0.092630, -0.039065]), NAR-identity -0.444508
- Llama-3.2-3B down_proj only: NAR-Hadamard -0.037185 (90% CI [-0.041228, -0.033142]), NAR-identity -0.267176
- Llama-3.2-1B q/k/v only: NAR-Hadamard -0.042843 (90% CI [-0.053356, -0.032330]), NAR-identity -0.370298
- Llama-3.2-1B both sites: NAR-Hadamard -0.226441 (90% CI [-0.256454, -0.196429]), NAR-identity -1.050347
- Llama-3.2-1B down_proj only: NAR-Hadamard -0.152464 (90% CI [-0.175749, -0.129180]), NAR-identity -0.635713
- Llama-3.1-8B q/k/v only: NAR-Hadamard -0.018431 (90% CI [-0.029940, -0.006921]), NAR-identity -0.214685
- Llama-3.1-8B both sites: NAR-Hadamard -0.061333 (90% CI [-0.066259, -0.056407]), NAR-identity -0.544747
- Llama-3.1-8B down_proj only: NAR-Hadamard -0.053245 (90% CI [-0.056954, -0.049536]), NAR-identity -0.311877

The maximum measured relative output discrepancy from storing the algebraically folded weights in bf16 was 0.00307489; this rounding is included in every rotated-method PPL result. No post-result tuning was performed.

## E6 — factorized R4 online cost

For the 3B down_proj input, G(V) is implemented as 64 sequential Householder reflections (below the 2k=128 cap), followed by a fixed permutation, signs, and block H128. The dense 8192x8192 matrix is materialized only for the equivalence audit, not the benchmark implementation.

| tokens | householder_reflections | nar_flops_per_token | down_matmul_flops_per_token | nar_flop_ratio_vs_matmul | nar_ms | hadamard_ms | down_matmul_ms | nar_wall_ratio_vs_hadamard | nar_wall_ratio_vs_down_matmul | nar_exceeds_10pct_matmul_wall |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 2162688 | 50331648 | 0.0429688 | 2.71507 | 0.303247 | 0.0186059 | 8.95332 | 145.925 | True |
| 32 | 64 | 2162688 | 50331648 | 0.0429688 | 3.29121 | 0.302085 | 0.0231595 | 10.895 | 142.111 | True |
| 2048 | 64 | 2162688 | 50331648 | 0.0429688 | 12.8809 | 2.61599 | 0.511313 | 4.92393 | 25.1919 | True |

Factorized-versus-dense fp32 verification: max absolute error 1.90735e-06, relative L2 error 3.65942e-07. **Engineering check: FAIL. The measured unfused wall-clock cost exceeds 10% of down_proj matmul cost.**

## E7 — per-token V cache under NAR

V is rotated within each KV head before dynamic asymmetric per-token INT4. The same R^T is folded blockwise into o_proj input columns (R2 position); the reported range/NMSE is offline and paired on identical V samples.

| b | method | layers | mean_group_range | mean_relative_quantization_error_nmse | mean_range_reduction_vs_hadamard | mean_nmse_delta_vs_hadamard |
|---|---|---|---|---|---|---|
| 32 | bf16 | 28 | 1.42223 | 0 | -0.0618682 | -0.00608396 |
| 32 | identity | 28 | 1.42223 | 0.00713789 | -0.0618682 | 0.00105393 |
| 32 | hadamard | 28 | 1.34693 | 0.00608396 | 0 | 0 |
| 32 | nar | 28 | 1.27817 | 0.00551637 | 0.0489613 | -0.000567584 |
| 64 | bf16 | 28 | 1.68782 | 0 | -0.118639 | -0.00795434 |
| 64 | identity | 28 | 1.68782 | 0.0103286 | -0.118639 | 0.00237422 |
| 64 | hadamard | 28 | 1.52233 | 0.00795434 | 0 | 0 |
| 64 | nar | 28 | 1.47109 | 0.0074602 | 0.0316668 | -0.000494145 |
| 128 | bf16 | 28 | 1.98227 | 0 | -0.190937 | -0.00983341 |
| 128 | identity | 28 | 1.98227 | 0.0144255 | -0.190937 | 0.00459208 |
| 128 | hadamard | 28 | 1.68325 | 0.00983341 | 0 | 0 |
| 128 | nar | 28 | 1.64935 | 0.00945025 | 0.0183523 | -0.000383152 |

The pooled fits `range(k)/range(0) = intercept + slope*sqrt(1-f)` are:

| model | b | intercept | slope | r_squared | rmse | points | fit |
|---|---|---|---|---|---|---|---|
| llama32_3b | 32 | 0.403757 | 0.59478 | 0.939523 | 0.00383579 | 140 | OLS range(k)/range(k=0) = intercept + slope*sqrt(1-f), pooled layer-k |
| llama32_3b | 64 | 0.385151 | 0.613609 | 0.950611 | 0.00271933 | 84 | OLS range(k)/range(k=0) = intercept + slope*sqrt(1-f), pooled layer-k |
| llama32_3b | 128 | 0.344142 | 0.655245 | 0.959319 | 0.00200938 | 56 | OLS range(k)/range(k=0) = intercept + slope*sqrt(1-f), pooled layer-k |

The o_proj fold identity has maximum fp64 absolute discrepancy 1.52095e-07.

![E7 V range versus k](results/activation/e7_v_range_vs_k.png)

## E8 — one-shot range-direct refinement

Starting from the frozen second-moment down_input V, each layer receives exactly 200 projected Riemannian-gradient steps with QR retraction, p=8, one seed, learning rate 0.05, and unit-Frobenius tangent normalization. Pi and signs remain fixed. Calibration uses cal-A samples; evaluation uses the next 128 disjoint WikiText-2 train chunks (cal-B). These choices were not changed after observing results.

| split | method | layers | mean_group_range | mean_relative_quantization_error_nmse | mean_range_delta_vs_second_moment | mean_nmse_delta_vs_second_moment |
|---|---|---|---|---|---|---|
| calibration_a | second_moment | 28 | 0.282987 | 0.00572961 | 0 | 0 |
| calibration_a | range_direct | 28 | 0.28044 | 0.00559949 | -0.00254689 | -0.000130127 |
| heldout_cal_b | second_moment | 28 | 0.286174 | 0.00591952 | 0 | 0 |
| heldout_cal_b | range_direct | 28 | 0.286734 | 0.00590786 | 0.000559774 | -1.16599e-05 |

Held-out sign: mean range did not improve (26/28 layers); mean NMSE improved (22/28 layers). This diagnostic is reported as-is.

![E8 held-out deltas](results/activation/e8_heldout_deltas.png)

## Artifact retention and protocol integrity

The original E1c q_input/down_input bf16 dumps remain untouched for E3/E4. E7 and E8 add separate V-cal-A and down-cal-B sampled dumps under project storage; none of the raw dumps enter Git. All result tables, completion metadata, factorization audits, plots, Slurm transcripts, and commands are published. E1/E2/E1c results were not rerun or modified.

# E11 — fair baselines in the E5 setting

The E5 bf16, both-site Hadamard, and both-site NAR rows are reused verbatim. New rows use the identical 64 WikiText-2 test chunks and three paired rotation seeds; only post-RMSNorm q/k/v inputs and down_proj inputs are fake-quantized. There was no sweep or post-result tuning.

| model | method | mean_ppl | ppl_delta_vs_bf16 | effective_bits_qkv | effective_bits_down | delta_vs_Had_90CI | delta_vs_NAR_90CI |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | bf16 | 7.61682 | 0 | 16 | 16 | N/A | N/A |
| Llama-3.2-3B | Hadamard, asym g128 (E5) | 7.77113 | 0.154305 | 4.25 | 4.25 | 0.000000 [0.000000, 0.000000] | 0.065848 [0.039065, 0.092630] |
| Llama-3.2-3B | NAR, asym g128, kmax (E5) | 7.70528 | 0.0884571 | 4.25 | 4.25 | -0.065848 [-0.092630, -0.039065] | 0.000000 [0.000000, 0.000000] |
| Llama-3.2-3B | SmoothQuant + Hadamard, asym g128 | 7.79104 | 0.174213 | 4.25 | 4.25 | 0.019908 [0.012168, 0.027649] | 0.085756 [0.066702, 0.104811] |
| Llama-3.2-3B | DuQuant-style, asym g128 | 7.75416 | 0.137335 | 4.25 | 4.25 | -0.016970 [-0.033445, -0.000494] | 0.048878 [0.036346, 0.061410] |
| Llama-3.2-3B | Hadamard, symmetric per-token | 8.01395 | 0.397129 | 4.00521 | 4.00195 | 0.242824 [0.236481, 0.249167] | 0.308672 [0.276337, 0.341007] |
| Llama-3.2-3B | Hadamard, asymmetric per-token | 7.91562 | 0.298799 | 4.01042 | 4.00391 | 0.144494 [0.131128, 0.157861] | 0.210342 [0.185714, 0.234970] |
| Llama-3.2-3B | NAR, asym g64, kmax | 7.68077 | 0.0639417 | 4.5 | 4.5 | -0.090363 [-0.101278, -0.079449] | -0.024515 [-0.041228, -0.007803] |
| Llama-3.2-3B | NAR, asym g256, kmax | 7.72464 | 0.107813 | 4.125 | 4.125 | -0.046492 [-0.050932, -0.042051] | 0.019356 [-0.003931, 0.042643] |
| Llama-3.2-3B | NAR, asym g128, k=8 | 7.71159 | 0.094766 | 4.25 | 4.25 | -0.059539 [-0.072446, -0.046632] | 0.006309 [-0.007644, 0.020262] |
| Llama-3.2-3B | NAR, asym g128, k=16 | 7.71202 | 0.0952 | 4.25 | 4.25 | -0.059105 [-0.063637, -0.054573] | 0.006743 [-0.015614, 0.029099] |
| Llama-3.2-3B | NAR, asym g128, k=32 | 7.70651 | 0.0896842 | 4.25 | 4.25 | -0.064621 [-0.075694, -0.053547] | 0.001227 [-0.014526, 0.016980] |
| Llama-3.1-8B | bf16 | 6.20401 | 0 | 16 | 16 | N/A | N/A |
| Llama-3.1-8B | Hadamard, asym g128 (E5) | 6.34626 | 0.142254 | 4.25 | 4.25 | 0.000000 [0.000000, 0.000000] | 0.061333 [0.056407, 0.066259] |
| Llama-3.1-8B | NAR, asym g128, kmax (E5) | 6.28493 | 0.0809206 | 4.25 | 4.25 | -0.061333 [-0.066259, -0.056407] | 0.000000 [0.000000, 0.000000] |
| Llama-3.1-8B | SmoothQuant + Hadamard, asym g128 | 6.35792 | 0.153913 | 4.25 | 4.25 | 0.011659 [0.008521, 0.014797] | 0.072992 [0.066167, 0.079817] |
| Llama-3.1-8B | DuQuant-style, asym g128 | 6.33393 | 0.129926 | 4.25 | 4.25 | -0.012327 [-0.016047, -0.008608] | 0.049006 [0.041183, 0.056828] |
| Llama-3.1-8B | Hadamard, symmetric per-token | 6.61061 | 0.406607 | 4.00391 | 4.00112 | 0.264353 [0.257897, 0.270808] | 0.325686 [0.314309, 0.337063] |
| Llama-3.1-8B | Hadamard, asymmetric per-token | 6.51651 | 0.312501 | 4.00781 | 4.00223 | 0.170247 [0.166250, 0.174245] | 0.231581 [0.226431, 0.236730] |
| Llama-3.1-8B | NAR, asym g64, kmax | 6.26141 | 0.0574021 | 4.5 | 4.5 | -0.084852 [-0.085437, -0.084266] | -0.023518 [-0.028449, -0.018588] |
| Llama-3.1-8B | NAR, asym g256, kmax | 6.30517 | 0.101158 | 4.125 | 4.125 | -0.041096 [-0.050918, -0.031273] | 0.020238 [0.014311, 0.026164] |
| Llama-3.1-8B | NAR, asym g128, k=8 | 6.29571 | 0.0917042 | 4.25 | 4.25 | -0.050550 [-0.054498, -0.046601] | 0.010784 [0.003442, 0.018125] |
| Llama-3.1-8B | NAR, asym g128, k=16 | 6.29542 | 0.0914089 | 4.25 | 4.25 | -0.050845 [-0.056853, -0.044837] | 0.010488 [0.008392, 0.012584] |
| Llama-3.1-8B | NAR, asym g128, k=32 | 6.28904 | 0.0850314 | 4.25 | 4.25 | -0.057222 [-0.062285, -0.052160] | 0.004111 [-0.005529, 0.013751] |

Effective activation bits/value include fp16 metadata: asymmetric group-g uses 4 + 32/g bits (one fp16 scale and one fp16 real-valued zero-point per group); symmetric per-token uses 4 + 16/n; asymmetric per-token uses 4 + 32/n. SmoothQuant channel scales are statically folded into bf16 weights and therefore add no per-token metadata.

## Baseline construction audit

SmoothQuant uses the fixed standard alpha=0.5 rule s_c=max|x_c|^0.5/max|w_c|^0.5, applies x/s and W*s, and then the same random-sign full Hadamard.

The DuQuant-style row was implemented after reading the [pinned official code](https://github.com/Hsu1023/DuQuant/tree/d56cfc6fe97c34c0eb100fec82fe439865905679) and the [NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/9febda1c8344cc5f2d51713964864e93-Paper-Conference.pdf). It uses the official zigzag distribution based on calibration-channel absolute maxima; within every resulting block of 128 it maps the single largest channel row to the uniform direction and uses a seeded random orthogonal basis on the complement. The official implementation can apply a greedy multi-step rotation, retain the prefix minimizing range, permute, and apply a second greedy rotation. The requested fair row deliberately omits that global multi-step prefix and second post-permutation rotation. Relative to NAR it aligns one greedy channel per block rather than top-k second-moment eigen-directions and has no explicit DC/zero-point alignment.

For Llama-3.2-3B q/k/v, requested k=32 is dimension-capped to the 24 available group-128 DC slots; all other reported k values are realized as requested. E11 calibration used the same 128 sequences, three-pass randomized eigensolver, and stride-32 permutation-energy sample. Full eigenvalue/energy/Ritz-residual CSVs are retained; transient eigenvector checkpoints were discarded after factor construction to respect project quota.

## Stop decision

The pre-registered stop condition did not fire: neither SmoothQuant+Hadamard nor the DuQuant-style row matches NAR within the paired 90% CI on either model. The decision is based on baseline-minus-NAR PPL with a paired two-sided 90% Student-t CI over the three seeds; a lower bound <= 0 denotes compatible-or-better and triggers the stop.

The maximum measured relative bf16 weight-fold discrepancy across the E11 rows is 0.0106346. Negative rows and engineering failures are retained.

# E12 — compact-WY deployable R4

The sequential Householder product is represented exactly in compact WY form as G=I-WY^T, with W,Y in R^(8192 x k). Applying G therefore uses two small matrix multiplications; the complete R4 then applies the fixed permutation/sign and block H128. Results cover the E11 knee ranks k=16/32 and the original k=64. The unfused Hadamard reference is the same staged PyTorch FWHT as E6, not a custom fused kernel. Rotation timing is fp32 and down_proj timing is bf16, matching E6.

| k | wy_vs_sequential_g_max_abs_error | wy_vs_sequential_g_relative_l2_error | wy_full_vs_dense_max_abs_error | wy_full_vs_dense_relative_l2_error |
|---|---|---|---|---|
| 16 | 2.38419e-06 | 1.09899e-07 | 1.43051e-06 | 2.8451e-07 |
| 32 | 2.38419e-06 | 1.51184e-07 | 1.78814e-06 | 2.91399e-07 |
| 64 | 2.38419e-06 | 2.13384e-07 | 1.66893e-06 | 3.10324e-07 |

| tokens | k | r4_flop_ratio_vs_matmul | wy_g_ms | r4_wy_ms | unfused_hadamard_ms | down_matmul_ms | r4_wall_ratio_vs_unfused_hadamard | r4_wall_ratio_vs_down_matmul | r4_under_10pct_matmul_wall |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0.0117188 | 0.0282944 | 0.2556 | 0.295857 | 0.0187339 | 0.863931 | 13.6437 | False |
| 32 | 16 | 0.0117188 | 0.0270965 | 0.255299 | 0.294578 | 0.0230624 | 0.86666 | 11.0699 | False |
| 2048 | 16 | 0.0117188 | 0.136342 | 2.0828 | 2.62036 | 0.552667 | 0.794852 | 3.76864 | False |
| 1 | 32 | 0.0221354 | 0.0271787 | 0.257383 | 0.298996 | 0.0185568 | 0.860825 | 13.87 | False |
| 32 | 32 | 0.0221354 | 0.0270997 | 0.257404 | 0.29579 | 0.0235264 | 0.870225 | 10.9411 | False |
| 2048 | 32 | 0.0221354 | 0.153205 | 2.09457 | 2.61418 | 0.526053 | 0.801235 | 3.98167 | False |
| 1 | 64 | 0.0429688 | 0.0256427 | 0.255199 | 0.29749 | 0.018576 | 0.85784 | 13.7381 | False |
| 32 | 64 | 0.0429688 | 0.0272139 | 0.256044 | 0.296311 | 0.0233952 | 0.864103 | 10.9443 | False |
| 2048 | 64 | 0.0429688 | 0.202283 | 2.13606 | 2.60901 | 0.524154 | 0.818726 | 4.07526 | False |

**2048-token engineering gate: FAIL.** At least one measured compact-WY rank exceeds 10% of down_proj wall-clock.

At one token the WY form is not under 10%; this regime is launch-bound. The unfused Hadamard has the same qualitative problem, so the single-token result is reported rather than hidden.

The FLOP ratio includes both WY matmuls plus the block-Hadamard/sign term; wall-clock ratios include the entire R4. No fused kernel or timing-specific tuning was used.

# E13 — zero-shot accuracy transfer

The pinned lm-evaluation-harness is commit b954108c9baaaa934b4ad842033b31a97ee30816. All rows are zero-shot and use seed 20260902, the same task examples, prompts, tokenizer, and metric definitions. PIQA, ARC-e, ARC-c, and HellaSwag use normalized accuracy; WinoGrande and LAMBADA use accuracy. The mean is the unweighted mean of those six values. bf16, Hadamard, and NAR use the E5 both-site activation-only setting; this one-seed transfer check has no confidence interval.

| model | method | task | metric | accuracy | delta_vs_bf16 |
|---|---|---|---|---|---|
| Llama-3.2-3B | bf16 | piqa | acc_norm,none | 0.778564 | 0 |
| Llama-3.2-3B | bf16 | arc_easy | acc_norm,none | 0.720539 | 0 |
| Llama-3.2-3B | bf16 | arc_challenge | acc_norm,none | 0.461604 | 0 |
| Llama-3.2-3B | bf16 | hellaswag | acc_norm,none | 0.741287 | 0 |
| Llama-3.2-3B | bf16 | winogrande | acc,none | 0.693765 | 0 |
| Llama-3.2-3B | bf16 | lambada_openai | acc,none | 0.69804 | 0 |
| Llama-3.2-3B | bf16 | mean | unweighted_mean_selected_accuracy | 0.6823 | 0 |
| Llama-3.2-3B | hadamard | piqa | acc_norm,none | 0.773667 | -0.00489663 |
| Llama-3.2-3B | hadamard | arc_easy | acc_norm,none | 0.718855 | -0.0016835 |
| Llama-3.2-3B | hadamard | arc_challenge | acc_norm,none | 0.457338 | -0.00426621 |
| Llama-3.2-3B | hadamard | hellaswag | acc_norm,none | 0.735312 | -0.00597491 |
| Llama-3.2-3B | hadamard | winogrande | acc,none | 0.684294 | -0.00947119 |
| Llama-3.2-3B | hadamard | lambada_openai | acc,none | 0.685232 | -0.0128081 |
| Llama-3.2-3B | hadamard | mean | unweighted_mean_selected_accuracy | 0.675783 | -0.00651675 |
| Llama-3.2-3B | nar | piqa | acc_norm,none | 0.767682 | -0.0108814 |
| Llama-3.2-3B | nar | arc_easy | acc_norm,none | 0.713805 | -0.00673401 |
| Llama-3.2-3B | nar | arc_challenge | acc_norm,none | 0.455631 | -0.0059727 |
| Llama-3.2-3B | nar | hellaswag | acc_norm,none | 0.737901 | -0.00338578 |
| Llama-3.2-3B | nar | winogrande | acc,none | 0.68824 | -0.00552486 |
| Llama-3.2-3B | nar | lambada_openai | acc,none | 0.692412 | -0.00562779 |
| Llama-3.2-3B | nar | mean | unweighted_mean_selected_accuracy | 0.675945 | -0.00635442 |
| Llama-3.1-8B | bf16 | piqa | acc_norm,none | 0.806311 | 0 |
| Llama-3.1-8B | bf16 | arc_easy | acc_norm,none | 0.825337 | 0 |
| Llama-3.1-8B | bf16 | arc_challenge | acc_norm,none | 0.546075 | 0 |
| Llama-3.1-8B | bf16 | hellaswag | acc_norm,none | 0.793368 | 0 |
| Llama-3.1-8B | bf16 | winogrande | acc,none | 0.743489 | 0 |
| Llama-3.1-8B | bf16 | lambada_openai | acc,none | 0.747332 | 0 |
| Llama-3.1-8B | bf16 | mean | unweighted_mean_selected_accuracy | 0.743652 | 0 |
| Llama-3.1-8B | hadamard | piqa | acc_norm,none | 0.803047 | -0.00326442 |
| Llama-3.1-8B | hadamard | arc_easy | acc_norm,none | 0.822811 | -0.00252525 |
| Llama-3.1-8B | hadamard | arc_challenge | acc_norm,none | 0.53413 | -0.0119454 |
| Llama-3.1-8B | hadamard | hellaswag | acc_norm,none | 0.78799 | -0.00537741 |
| Llama-3.1-8B | hadamard | winogrande | acc,none | 0.7206 | -0.0228887 |
| Llama-3.1-8B | hadamard | lambada_openai | acc,none | 0.742092 | -0.00523967 |
| Llama-3.1-8B | hadamard | mean | unweighted_mean_selected_accuracy | 0.735112 | -0.00854014 |
| Llama-3.1-8B | nar | piqa | acc_norm,none | 0.804679 | -0.00163221 |
| Llama-3.1-8B | nar | arc_easy | acc_norm,none | 0.818182 | -0.00715488 |
| Llama-3.1-8B | nar | arc_challenge | acc_norm,none | 0.544369 | -0.00170648 |
| Llama-3.1-8B | nar | hellaswag | acc_norm,none | 0.788588 | -0.00477992 |
| Llama-3.1-8B | nar | winogrande | acc,none | 0.741121 | -0.0023678 |
| Llama-3.1-8B | nar | lambada_openai | acc,none | 0.747138 | -0.000194062 |
| Llama-3.1-8B | nar | mean | unweighted_mean_selected_accuracy | 0.740679 | -0.00297256 |

Paired aggregate transfer: Llama-3.2-3B: NAR-Hadamard mean-accuracy delta +0.000162; Llama-3.1-8B: NAR-Hadamard mean-accuracy delta +0.005568.

These accuracy results are reported regardless of sign. No task subset, prompt, batch-size, or metric was selected after observing outputs.

# E14 — end-to-end W4A4KV4 (in progress)

## Released-code sanity anchor

The official released QuaRot W4A4KV4 pipeline produced WikiText-2 PPL **6.355** for Llama-2-7B, versus the published **6.10** target (absolute error **0.255**; requested tolerance ±0.10). The sanity anchor is therefore a recorded **FAIL**. Per the explicit project decision, this release/paper discrepancy is retained as a negative reproducibility result and the frozen 3B/8B experiment matrix proceeds without reclassifying the anchor. No anchor result was tuned or rerun after this decision.

## Single-seed protocol amendment

E14 was explicitly amended to one seed per configuration to accelerate completion. Seed 0 is the sole planned E14 seed; partial seed-1 work is excluded from the headline comparison. Direct paired deltas are reported without seed-level confidence intervals. Existing completed E5/E11 multi-seed results are unchanged.

| model | seed | row | full WikiText-2 PPL | paired delta vs Hadamard |
|---|---|---|---|---|
| Llama-3.2-3B | 0 | Hadamard + asymmetric g128 | 9.20901 | 0 |
| Llama-3.2-3B | 0 | NAR k=8 R1/R4 + NAR R2 + asymmetric g128 | 8.75623 | -0.452776 |
| Llama-3.2-3B | 0 | NAR k=max R1/R4 + NAR R2 + asymmetric g128 | 8.71445 | -0.494562 |
| Llama-3.2-3B | 0 | QuaRot released symmetric A4 | 10.33237 | +1.12336 |

Completed six-task zero-shot means are 0.643694 for metadata-matched Hadamard and 0.651201 for NAR k=8, a direct paired gain of +0.007507 for NAR. The released QuaRot zero-shot row is still running.
