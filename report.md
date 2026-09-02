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
