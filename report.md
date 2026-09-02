# NAR offline tensor validation

## Outcome

- **Phenomenon criterion: FAIL**. At b=128, NAR mean K range was 7.770% below full-head Hadamard. The operational top-k projection attribution was 15.231%, and a rotation recomputed on disjoint calibration data retained 100.914% of the calibration-A reduction.
- **E2 PPL gate: PASS**. The paired 3B b=128 NAR-Hadamard PPL delta was -0.017134, with paired 90% CI [-0.027894, -0.006374].
- **Method-promising criterion: FAIL**. This requires both the phenomenon criterion and the additional E2 PPL gate.
- **E0 implementation sanity: PASS**.
- QK invariance over every valid layer/method check: maximum absolute error 7.62939e-05, maximum relative error 4.61466e-07.

## Fixed protocol and quantizer semantics

The fake quantizer is dynamic asymmetric per token and contiguous channel group. For each group, `s=(max-min)/15` and `z=min`; both `s` and the real-valued offset `z` are rounded to IEEE fp16 before use. It then computes `q=clamp(round((x-z)/s),0,15)` and `x_hat=q*s+z`. Relative quantization error is NMSE `sum((x_hat-x)^2)/sum(x^2)`. Degenerate groups use scale 1 and reproduce their fp16 offset. All comparisons are paired on identical tensors. No GPTQ, weight quantization, or end-to-end W4A4 pipeline was run.

Primary E1 uses 128 WikiText-2 train sequences of length 2048 with BOS prepended. Uncentered second moments use every token and KV head. Offline tensor dumps store positions 0,32,...,2016 from every sequence, all 24 Q heads, and all 8 KV heads; range analysis uses K. Calibration B is the next disjoint 128 train sequences. NAR is layer-specific and K-only calibrated; Q is never included.

The pre-registered stability interpretation is positive held-out reduction and at least 80% retention of the calibration-A reduction. Because 'substantial share' had no numeric threshold in the prompt, this report operationally uses 10%, matching the primary range threshold.

## Compute actually used

| stage | Slurm job | node | GPU | seconds |
|---|---|---|---|---|
| E1 cal A | 135864 | gpu-6000ada-1.cluster02.eee.ntu.edu.sg | NVIDIA RTX 6000 Ada Generation | 17.3648 |
| E1 cal B | 135840 | gpu-6000ada-1.cluster02.eee.ntu.edu.sg | NVIDIA RTX 6000 Ada Generation | 15.4152 |
| E2 llama32_3b | 135840 | gpu-6000ada-1.cluster02.eee.ntu.edu.sg | NVIDIA RTX 6000 Ada Generation | 184.367 |
| E2 llama32_1b | 135840 | gpu-6000ada-1.cluster02.eee.ntu.edu.sg | NVIDIA RTX 6000 Ada Generation | 43.07 |

## E0 — synthetic sanity

| b | Hadamard range | 2|x|/sqrt(b) | NAR range | pass |
|---|---|---|---|---|
| 32 | 35.3553 | 35.3553 | 0 | True |
| 64 | 25 | 25 | 0 | True |
| 128 | 17.6777 | 17.6777 | 0 | True |

Constant-shift checks (the tiny nonzero MSE deltas, if present, are solely from required fp16 metadata rounding):

| b | c | |range delta| | base MSE | shifted MSE | |MSE delta| | pass |
|---|---|---|---|---|---|---|
| 32 | -8 | 0 | 0.0059784 | 0.00597934 | 9.46689e-07 | True |
| 32 | -1 | 0 | 0.0059784 | 0.00597893 | 5.32717e-07 | True |
| 32 | 0 | 0 | 0.0059784 | 0.0059784 | 0 | True |
| 32 | 1 | 0 | 0.0059784 | 0.0059789 | 5.04311e-07 | True |
| 32 | 8 | 0 | 0.0059784 | 0.00597981 | 1.41608e-06 | True |
| 64 | -8 | 0 | 0.00797618 | 0.00797485 | 1.33179e-06 | True |
| 64 | -1 | 0 | 0.00797618 | 0.00797636 | 1.74157e-07 | True |
| 64 | 0 | 0 | 0.00797618 | 0.00797618 | 0 | True |
| 64 | 1 | 0 | 0.00797618 | 0.00797623 | 4.37722e-08 | True |
| 64 | 8 | 0 | 0.00797618 | 0.0079774 | 1.2219e-06 | True |
| 128 | -8 | 0 | 0.00981289 | 0.00981458 | 1.68663e-06 | True |
| 128 | -1 | 0 | 0.00981289 | 0.00981358 | 6.88247e-07 | True |
| 128 | 0 | 0 | 0.00981289 | 0.00981289 | 0 | True |
| 128 | 1 | 0 | 0.00981289 | 0.00981333 | 4.42378e-07 | True |
| 128 | 8 | 0 | 0.00981289 | 0.00981409 | 1.19675e-06 | True |

## E1 — averaged results

| b | method | mean K range | relative quantization error (NMSE) |
|---|---|---|---|
| 128 | bf16 | 15.8428 | 0 |
| 128 | hadamard | 9.60617 | 0.00941465 |
| 128 | identity | 15.8428 | 0.0261339 |
| 128 | nar | 8.85975 | 0.00802749 |
| 128 | nar_rope | N/A | N/A |
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

NAR-RoPE at b=128 is N/A by construction: head_dim=128 and b=128 provide only one group, while an invariant RoPE plane has two basis directions that the stated method requires mapping to two different group DCs. No surrogate value was inserted.

### Low-rank check

| b | k | mean NAR K range | DC absorption vs k=0 | projection range attribution |
|---|---|---|---|---|
| 128 | 0 | 9.37147 | 0 | 0 |
| 128 | 1 | 8.85975 | 0.0547613 | 0.152311 |
| 32 | 0 | 7.18035 | 0 | 0 |
| 32 | 1 | 6.87043 | 0.0433439 | 0.091374 |
| 32 | 2 | 6.60709 | 0.0800279 | 0.159911 |
| 32 | 3 | 6.36081 | 0.114322 | 0.216978 |
| 32 | 4 | 6.14034 | 0.145086 | 0.269542 |
| 64 | 0 | 8.27888 | 0 | 0 |
| 64 | 1 | 7.87514 | 0.0489393 | 0.121988 |
| 64 | 2 | 7.56239 | 0.0867687 | 0.206672 |

| eigen rank | mean trace fraction | mean cumulative fraction |
|---|---|---|
| 1 | 0.122013 | 0.122013 |
| 128 | 0.00115555 | 1 |
| 16 | 0.00736323 | 0.596915 |
| 2 | 0.0890583 | 0.211072 |
| 32 | 0.00556534 | 0.696694 |
| 4 | 0.0653333 | 0.352597 |
| 64 | 0.0038016 | 0.843498 |
| 8 | 0.0291664 | 0.522792 |

Mean effective rank was 23.1100/128; mean top-1 energy fraction was 12.2013%.

![Mean K range vs k](results/llama32_3b/range_vs_k.png)

![Eigenvalue spectrum](results/llama32_3b/eigenvalue_spectrum.png)

### Stability

| b | A reduction vs Had | disjoint-B reduction vs Had | degradation (fraction points) | retained A reduction |
|---|---|---|---|---|
| 128 | 0.0780328 | 0.0784117 | -0.000378935 | 1.00914 |
| 32 | 0.205378 | 0.20518 | 0.000198085 | 0.998999 |
| 64 | 0.12656 | 0.125746 | 0.000813894 | 0.993069 |

### NAR-RoPE position check

The same stored pre-RoPE K sample is rotated with the model's exact cos/sin at each requested hypothetical position; this isolates positional rotation from token-distribution changes.

| b | position | method | mean range | reduction vs Had |
|---|---|---|---|---|
| 128 | 0 | bf16 | 15.8773 | N/A |
| 128 | 0 | hadamard | 9.6248 | 0 |
| 128 | 0 | nar_rope | N/A | N/A |
| 128 | 1024 | bf16 | 15.8578 | N/A |
| 128 | 1024 | hadamard | 9.59643 | 0 |
| 128 | 1024 | nar_rope | N/A | N/A |
| 128 | 2048 | bf16 | 15.8363 | N/A |
| 128 | 2048 | hadamard | 9.6076 | 0 |
| 128 | 2048 | nar_rope | N/A | N/A |
| 128 | 512 | bf16 | 15.8551 | N/A |
| 128 | 512 | hadamard | 9.60327 | 0 |
| 128 | 512 | nar_rope | N/A | N/A |
| 32 | 0 | bf16 | 9.00604 | N/A |
| 32 | 0 | hadamard | 7.73433 | 0 |
| 32 | 0 | nar_rope | 6.95545 | 0.100789 |
| 32 | 1024 | bf16 | 8.96045 | N/A |
| 32 | 1024 | hadamard | 7.74547 | 0 |
| 32 | 1024 | nar_rope | 6.94879 | 0.102944 |
| 32 | 2048 | bf16 | 8.94702 | N/A |
| 32 | 2048 | hadamard | 7.71534 | 0 |
| 32 | 2048 | nar_rope | 6.95928 | 0.0981812 |
| 32 | 512 | bf16 | 8.931 | N/A |
| 32 | 512 | hadamard | 7.73475 | 0 |
| 32 | 512 | nar_rope | 6.95351 | 0.100975 |
| 64 | 0 | bf16 | 12.2065 | N/A |
| 64 | 0 | hadamard | 8.64831 | 0 |
| 64 | 0 | nar_rope | 8.15314 | 0.0571776 |
| 64 | 1024 | bf16 | 12.162 | N/A |
| 64 | 1024 | hadamard | 8.65966 | 0 |
| 64 | 1024 | nar_rope | 8.13601 | 0.0606517 |
| 64 | 2048 | bf16 | 12.1451 | N/A |
| 64 | 2048 | hadamard | 8.66659 | 0 |
| 64 | 2048 | nar_rope | 8.14638 | 0.0598483 |
| 64 | 512 | bf16 | 12.1381 | N/A |
| 64 | 512 | hadamard | 8.64453 | 0 |
| 64 | 512 | nar_rope | 8.15968 | 0.0561416 |

### Per-layer tables

#### b=32

| layer | method | mean range | NMSE | valid |
|---|---|---|---|---|
| 0 | bf16 | 7.11253 | 0 | True |
| 0 | identity | 7.11253 | 0.00850928 | True |
| 0 | hadamard | 6.26104 | 0.00575211 | True |
| 0 | nar | 4.70187 | 0.00320179 | True |
| 0 | nar_rope | 5.58033 | 0.00453177 | True |
| 1 | bf16 | 8.71763 | 0 | True |
| 1 | identity | 8.71763 | 0.00790949 | True |
| 1 | hadamard | 7.7165 | 0.00575411 | True |
| 1 | nar | 6.02154 | 0.00357953 | True |
| 1 | nar_rope | 7.06256 | 0.00486979 | True |
| 2 | bf16 | 8.3854 | 0 | True |
| 2 | identity | 8.3854 | 0.00924123 | True |
| 2 | hadamard | 7.34703 | 0.00592595 | True |
| 2 | nar | 5.76221 | 0.00371782 | True |
| 2 | nar_rope | 6.45097 | 0.0046784 | True |
| 3 | bf16 | 9.01736 | 0 | True |
| 3 | identity | 9.01736 | 0.0102791 | True |
| 3 | hadamard | 7.37495 | 0.00560782 | True |
| 3 | nar | 5.67194 | 0.00341847 | True |
| 3 | nar_rope | 6.32486 | 0.00421155 | True |
| 4 | bf16 | 8.96766 | 0 | True |
| 4 | identity | 8.96766 | 0.00969337 | True |
| 4 | hadamard | 7.43309 | 0.00572741 | True |
| 4 | nar | 5.81615 | 0.00356221 | True |
| 4 | nar_rope | 6.56828 | 0.00458945 | True |
| 5 | bf16 | 8.79107 | 0 | True |
| 5 | identity | 8.79107 | 0.0096937 | True |
| 5 | hadamard | 7.52558 | 0.00602197 | True |
| 5 | nar | 5.93389 | 0.00377456 | True |
| 5 | nar_rope | 6.71967 | 0.00485412 | True |
| 6 | bf16 | 8.90607 | 0 | True |
| 6 | identity | 8.90607 | 0.0085011 | True |
| 6 | hadamard | 7.79153 | 0.00584918 | True |
| 6 | nar | 6.29224 | 0.00386976 | True |
| 6 | nar_rope | 7.12649 | 0.00502825 | True |
| 7 | bf16 | 9.92257 | 0 | True |
| 7 | identity | 9.92257 | 0.00965142 | True |
| 7 | hadamard | 8.26439 | 0.0058269 | True |
| 7 | nar | 6.57121 | 0.00378789 | True |
| 7 | nar_rope | 7.15102 | 0.00448923 | True |
| 8 | bf16 | 9.46209 | 0 | True |
| 8 | identity | 9.46209 | 0.00928715 | True |
| 8 | hadamard | 8.14543 | 0.00597473 | True |
| 8 | nar | 6.50194 | 0.00392659 | True |
| 8 | nar_rope | 7.2467 | 0.00487745 | True |
| 9 | bf16 | 8.8653 | 0 | True |
| 9 | identity | 8.8653 | 0.00867272 | True |
| 9 | hadamard | 7.86712 | 0.00585676 | True |
| 9 | nar | 6.4616 | 0.0040084 | True |
| 9 | nar_rope | 7.10805 | 0.00492286 | True |
| 10 | bf16 | 8.89306 | 0 | True |
| 10 | identity | 8.89306 | 0.00939331 | True |
| 10 | hadamard | 7.66154 | 0.00601213 | True |
| 10 | nar | 6.31648 | 0.00414859 | True |
| 10 | nar_rope | 7.03606 | 0.00514828 | True |
| 11 | bf16 | 10.2085 | 0 | True |
| 11 | identity | 10.2085 | 0.0102216 | True |
| 11 | hadamard | 8.25848 | 0.00575277 | True |
| 11 | nar | 6.63522 | 0.00379088 | True |
| 11 | nar_rope | 7.20488 | 0.00461395 | True |
| 12 | bf16 | 8.5022 | 0 | True |
| 12 | identity | 8.5022 | 0.00818896 | True |
| 12 | hadamard | 7.91147 | 0.00607203 | True |
| 12 | nar | 6.51185 | 0.00415807 | True |
| 12 | nar_rope | 7.28115 | 0.00520057 | True |
| 13 | bf16 | 9.19267 | 0 | True |
| 13 | identity | 9.19267 | 0.00845793 | True |
| 13 | hadamard | 8.29599 | 0.00598135 | True |
| 13 | nar | 6.68122 | 0.00391965 | True |
| 13 | nar_rope | 7.55512 | 0.00500033 | True |
| 14 | bf16 | 9.0253 | 0 | True |
| 14 | identity | 9.0253 | 0.00888293 | True |
| 14 | hadamard | 8.09078 | 0.00605209 | True |
| 14 | nar | 6.37794 | 0.00380084 | True |
| 14 | nar_rope | 7.26848 | 0.00490283 | True |
| 15 | bf16 | 8.60356 | 0 | True |
| 15 | identity | 8.60356 | 0.00847519 | True |
| 15 | hadamard | 7.75355 | 0.0060116 | True |
| 15 | nar | 6.39664 | 0.0041553 | True |
| 15 | nar_rope | 7.19282 | 0.00526774 | True |
| 16 | bf16 | 9.00322 | 0 | True |
| 16 | identity | 9.00322 | 0.00893736 | True |
| 16 | hadamard | 8.0054 | 0.00601511 | True |
| 16 | nar | 6.39742 | 0.00391275 | True |
| 16 | nar_rope | 7.36562 | 0.00516783 | True |
| 17 | bf16 | 8.71669 | 0 | True |
| 17 | identity | 8.71669 | 0.00920372 | True |
| 17 | hadamard | 7.58286 | 0.00592199 | True |
| 17 | nar | 6.06058 | 0.00382036 | True |
| 17 | nar_rope | 6.88571 | 0.00496817 | True |
| 18 | bf16 | 8.96827 | 0 | True |
| 18 | identity | 8.96827 | 0.0091042 | True |
| 18 | hadamard | 7.77265 | 0.00599719 | True |
| 18 | nar | 6.23356 | 0.00390569 | True |
| 18 | nar_rope | 7.02547 | 0.00497801 | True |
| 19 | bf16 | 9.31478 | 0 | True |
| 19 | identity | 9.31478 | 0.00963938 | True |
| 19 | hadamard | 7.96236 | 0.00585718 | True |
| 19 | nar | 6.287 | 0.0037045 | True |
| 19 | nar_rope | 7.28616 | 0.00500509 | True |
| 20 | bf16 | 8.93745 | 0 | True |
| 20 | identity | 8.93745 | 0.00921841 | True |
| 20 | hadamard | 7.76962 | 0.00592193 | True |
| 20 | nar | 6.13409 | 0.00372851 | True |
| 20 | nar_rope | 6.98738 | 0.00488543 | True |
| 21 | bf16 | 8.97068 | 0 | True |
| 21 | identity | 8.97068 | 0.00926365 | True |
| 21 | hadamard | 7.58535 | 0.0058298 | True |
| 21 | nar | 6.04188 | 0.00376639 | True |
| 21 | nar_rope | 6.97555 | 0.00506044 | True |
| 22 | bf16 | 9.09305 | 0 | True |
| 22 | identity | 9.09305 | 0.0100741 | True |
| 22 | hadamard | 7.49041 | 0.00570522 | True |
| 22 | nar | 5.79508 | 0.00351739 | True |
| 22 | nar_rope | 6.65541 | 0.00461863 | True |
| 23 | bf16 | 8.97874 | 0 | True |
| 23 | identity | 8.97874 | 0.00922388 | True |
| 23 | hadamard | 7.83818 | 0.00597489 | True |
| 23 | nar | 6.21505 | 0.00377525 | True |
| 23 | nar_rope | 7.1596 | 0.00508961 | True |
| 24 | bf16 | 8.90708 | 0 | True |
| 24 | identity | 8.90708 | 0.00910962 | True |
| 24 | hadamard | 7.75472 | 0.00589292 | True |
| 24 | nar | 6.15464 | 0.00379123 | True |
| 24 | nar_rope | 6.97163 | 0.00482864 | True |
| 25 | bf16 | 9.51579 | 0 | True |
| 25 | identity | 9.51579 | 0.0105573 | True |
| 25 | hadamard | 7.62697 | 0.00555605 | True |
| 25 | nar | 5.92422 | 0.00336568 | True |
| 25 | nar_rope | 6.76123 | 0.00450873 | True |
| 26 | bf16 | 8.30226 | 0 | True |
| 26 | identity | 8.30226 | 0.00845245 | True |
| 26 | hadamard | 7.50961 | 0.00604894 | True |
| 26 | nar | 6.06026 | 0.00396034 | True |
| 26 | nar_rope | 6.90574 | 0.00518146 | True |
| 27 | bf16 | 8.92647 | 0 | True |
| 27 | identity | 8.92647 | 0.00946914 | True |
| 27 | hadamard | 7.63744 | 0.00570833 | True |
| 27 | nar | 5.97176 | 0.00353789 | True |
| 27 | nar_rope | 6.66516 | 0.00446552 | True |

#### b=64

| layer | method | mean range | NMSE | valid |
|---|---|---|---|---|
| 0 | bf16 | 9.36937 | 0 | True |
| 0 | identity | 9.36937 | 0.0144364 | True |
| 0 | hadamard | 7.07638 | 0.00754154 | True |
| 0 | nar | 5.86949 | 0.00505022 | True |
| 0 | nar_rope | 6.52104 | 0.00632968 | True |
| 1 | bf16 | 11.1226 | 0 | True |
| 1 | identity | 11.1226 | 0.0125646 | True |
| 1 | hadamard | 8.74138 | 0.00756677 | True |
| 1 | nar | 7.40654 | 0.00550653 | True |
| 1 | nar_rope | 8.4378 | 0.00708111 | True |
| 2 | bf16 | 11.4825 | 0 | True |
| 2 | identity | 11.4825 | 0.0159205 | True |
| 2 | hadamard | 8.16912 | 0.00753177 | True |
| 2 | nar | 7.08991 | 0.00580832 | True |
| 2 | nar_rope | 7.62456 | 0.00661688 | True |
| 3 | bf16 | 12.2472 | 0 | True |
| 3 | identity | 12.2472 | 0.0182192 | True |
| 3 | hadamard | 8.23849 | 0.00719017 | True |
| 3 | nar | 6.9705 | 0.00531754 | True |
| 3 | nar_rope | 7.47374 | 0.00602155 | True |
| 4 | bf16 | 12.3565 | 0 | True |
| 4 | identity | 12.3565 | 0.0174484 | True |
| 4 | hadamard | 8.31246 | 0.00735027 | True |
| 4 | nar | 7.15937 | 0.00552765 | True |
| 4 | nar_rope | 7.75701 | 0.00658688 | True |
| 5 | bf16 | 11.9664 | 0 | True |
| 5 | identity | 11.9664 | 0.0168159 | True |
| 5 | hadamard | 8.39099 | 0.00766211 | True |
| 5 | nar | 7.28089 | 0.00580115 | True |
| 5 | nar_rope | 7.8672 | 0.00673419 | True |
| 6 | bf16 | 11.9691 | 0 | True |
| 6 | identity | 11.9691 | 0.0149201 | True |
| 6 | hadamard | 8.78026 | 0.00761954 | True |
| 6 | nar | 7.61685 | 0.0057934 | True |
| 6 | nar_rope | 8.42294 | 0.00709664 | True |
| 7 | bf16 | 13.4523 | 0 | True |
| 7 | identity | 13.4523 | 0.0168718 | True |
| 7 | hadamard | 9.26409 | 0.0075187 | True |
| 7 | nar | 8.13235 | 0.00592981 | True |
| 7 | nar_rope | 8.48005 | 0.0064528 | True |
| 8 | bf16 | 12.8943 | 0 | True |
| 8 | identity | 12.8943 | 0.0163014 | True |
| 8 | hadamard | 9.08555 | 0.00764709 | True |
| 8 | nar | 8.03657 | 0.00610447 | True |
| 8 | nar_rope | 8.5039 | 0.00676719 | True |
| 9 | bf16 | 11.979 | 0 | True |
| 9 | identity | 11.979 | 0.0152187 | True |
| 9 | hadamard | 8.85106 | 0.00761426 | True |
| 9 | nar | 7.94542 | 0.00617859 | True |
| 9 | nar_rope | 8.34022 | 0.00687707 | True |
| 10 | bf16 | 12.063 | 0 | True |
| 10 | identity | 12.063 | 0.0159504 | True |
| 10 | hadamard | 8.58078 | 0.00773757 | True |
| 10 | nar | 7.7237 | 0.00629166 | True |
| 10 | nar_rope | 8.15868 | 0.00703271 | True |
| 11 | bf16 | 14.0736 | 0 | True |
| 11 | identity | 14.0736 | 0.0185438 | True |
| 11 | hadamard | 9.1926 | 0.0073206 | True |
| 11 | nar | 8.1985 | 0.00591876 | True |
| 11 | nar_rope | 8.55526 | 0.00645004 | True |
| 12 | bf16 | 11.6004 | 0 | True |
| 12 | identity | 11.6004 | 0.0139368 | True |
| 12 | hadamard | 8.93706 | 0.00796172 | True |
| 12 | nar | 7.85826 | 0.0062219 | True |
| 12 | nar_rope | 8.51812 | 0.00724874 | True |
| 13 | bf16 | 12.7163 | 0 | True |
| 13 | identity | 12.7163 | 0.0149197 | True |
| 13 | hadamard | 9.28448 | 0.00768133 | True |
| 13 | nar | 8.13144 | 0.00592965 | True |
| 13 | nar_rope | 8.80086 | 0.00692126 | True |
| 14 | bf16 | 11.9068 | 0 | True |
| 14 | identity | 11.9068 | 0.0147854 | True |
| 14 | hadamard | 9.06084 | 0.00778336 | True |
| 14 | nar | 7.73976 | 0.00572406 | True |
| 14 | nar_rope | 8.36484 | 0.00669437 | True |
| 15 | bf16 | 11.5454 | 0 | True |
| 15 | identity | 11.5454 | 0.0144127 | True |
| 15 | hadamard | 8.69212 | 0.00776324 | True |
| 15 | nar | 7.78588 | 0.00626456 | True |
| 15 | nar_rope | 8.28339 | 0.0070699 | True |
| 16 | bf16 | 12.0924 | 0 | True |
| 16 | identity | 12.0924 | 0.0150827 | True |
| 16 | hadamard | 9.01315 | 0.00782343 | True |
| 16 | nar | 7.86322 | 0.00601987 | True |
| 16 | nar_rope | 8.50944 | 0.00701694 | True |
| 17 | bf16 | 11.7743 | 0 | True |
| 17 | identity | 11.7743 | 0.0157868 | True |
| 17 | hadamard | 8.5198 | 0.00765431 | True |
| 17 | nar | 7.44888 | 0.00593001 | True |
| 17 | nar_rope | 8.08939 | 0.00690849 | True |
| 18 | bf16 | 12.2372 | 0 | True |
| 18 | identity | 12.2372 | 0.0154915 | True |
| 18 | hadamard | 8.73143 | 0.00776402 | True |
| 18 | nar | 7.66898 | 0.00604087 | True |
| 18 | nar_rope | 8.27732 | 0.00703065 | True |
| 19 | bf16 | 12.5909 | 0 | True |
| 19 | identity | 12.5909 | 0.0163523 | True |
| 19 | hadamard | 8.95946 | 0.00761184 | True |
| 19 | nar | 7.89133 | 0.00589704 | True |
| 19 | nar_rope | 8.55093 | 0.00705324 | True |
| 20 | bf16 | 12.166 | 0 | True |
| 20 | identity | 12.166 | 0.0158997 | True |
| 20 | hadamard | 8.66872 | 0.00754503 | True |
| 20 | nar | 7.61942 | 0.00589421 | True |
| 20 | nar_rope | 8.30247 | 0.00700158 | True |
| 21 | bf16 | 12.4105 | 0 | True |
| 21 | identity | 12.4105 | 0.0165465 | True |
| 21 | hadamard | 8.44952 | 0.00741351 | True |
| 21 | nar | 7.56077 | 0.00596436 | True |
| 21 | nar_rope | 8.20205 | 0.00708838 | True |
| 22 | bf16 | 12.7355 | 0 | True |
| 22 | identity | 12.7355 | 0.0184044 | True |
| 22 | hadamard | 8.42847 | 0.00743351 | True |
| 22 | nar | 7.14657 | 0.00542477 | True |
| 22 | nar_rope | 7.88539 | 0.00658446 | True |
| 23 | bf16 | 12.3261 | 0 | True |
| 23 | identity | 12.3261 | 0.0161233 | True |
| 23 | hadamard | 8.69405 | 0.00752861 | True |
| 23 | nar | 7.74243 | 0.00598478 | True |
| 23 | nar_rope | 8.28798 | 0.00693424 | True |
| 24 | bf16 | 12.3917 | 0 | True |
| 24 | identity | 12.3917 | 0.0163813 | True |
| 24 | hadamard | 8.70921 | 0.00762959 | True |
| 24 | nar | 7.63649 | 0.00593282 | True |
| 24 | nar_rope | 8.15339 | 0.00674589 | True |
| 25 | bf16 | 13.3901 | 0 | True |
| 25 | identity | 13.3901 | 0.0192404 | True |
| 25 | hadamard | 8.526 | 0.00711397 | True |
| 25 | nar | 7.38778 | 0.00530649 | True |
| 25 | nar_rope | 7.94727 | 0.00627408 | True |
| 26 | bf16 | 11.1871 | 0 | True |
| 26 | identity | 11.1871 | 0.014402 | True |
| 26 | hadamard | 8.40157 | 0.00774327 | True |
| 26 | nar | 7.3711 | 0.00598916 | True |
| 26 | nar_rope | 8.06248 | 0.00718922 | True |
| 27 | bf16 | 12.0754 | 0 | True |
| 27 | identity | 12.0754 | 0.0164548 | True |
| 27 | hadamard | 8.54725 | 0.00732746 | True |
| 27 | nar | 7.46443 | 0.0056228 | True |
| 27 | nar_rope | 7.83967 | 0.00628743 | True |

#### b=128

| layer | method | mean range | NMSE | valid |
|---|---|---|---|---|
| 0 | bf16 | 12.0048 | 0 | True |
| 0 | identity | 12.0048 | 0.0232094 | True |
| 0 | hadamard | 7.79391 | 0.00924772 | True |
| 0 | nar | 7.0069 | 0.00734377 | True |
| 0 | nar_rope | N/A | N/A | False |
| 1 | bf16 | 13.8245 | 0 | True |
| 1 | identity | 13.8245 | 0.0193157 | True |
| 1 | hadamard | 9.71578 | 0.00946489 | True |
| 1 | nar | 8.80139 | 0.0077929 | True |
| 1 | nar_rope | N/A | N/A | False |
| 2 | bf16 | 14.638 | 0 | True |
| 2 | identity | 14.638 | 0.0246744 | True |
| 2 | hadamard | 9.10534 | 0.00939902 | True |
| 2 | nar | 8.33559 | 0.00795143 | True |
| 2 | nar_rope | N/A | N/A | False |
| 3 | bf16 | 17.226 | 0 | True |
| 3 | identity | 17.226 | 0.0326497 | True |
| 3 | hadamard | 9.23786 | 0.00910608 | True |
| 3 | nar | 8.2148 | 0.0073134 | True |
| 3 | nar_rope | N/A | N/A | False |
| 4 | bf16 | 15.955 | 0 | True |
| 4 | identity | 15.955 | 0.0271839 | True |
| 4 | hadamard | 9.22147 | 0.0091177 | True |
| 4 | nar | 8.38974 | 0.00763435 | True |
| 4 | nar_rope | N/A | N/A | False |
| 5 | bf16 | 15.5505 | 0 | True |
| 5 | identity | 15.5505 | 0.0273059 | True |
| 5 | hadamard | 9.3986 | 0.00969368 | True |
| 5 | nar | 8.42453 | 0.00777265 | True |
| 5 | nar_rope | N/A | N/A | False |
| 6 | bf16 | 15.4317 | 0 | True |
| 6 | identity | 15.4317 | 0.0239034 | True |
| 6 | hadamard | 9.69903 | 0.00937202 | True |
| 6 | nar | 8.95312 | 0.0080027 | True |
| 6 | nar_rope | N/A | N/A | False |
| 7 | bf16 | 18.4491 | 0 | True |
| 7 | identity | 18.4491 | 0.0300999 | True |
| 7 | hadamard | 10.1924 | 0.00919608 | True |
| 7 | nar | 9.50612 | 0.00813698 | True |
| 7 | nar_rope | N/A | N/A | False |
| 8 | bf16 | 16.3675 | 0 | True |
| 8 | identity | 16.3675 | 0.0253718 | True |
| 8 | hadamard | 10.1242 | 0.00959983 | True |
| 8 | nar | 9.37017 | 0.00820091 | True |
| 8 | nar_rope | N/A | N/A | False |
| 9 | bf16 | 16.0702 | 0 | True |
| 9 | identity | 16.0702 | 0.0261528 | True |
| 9 | hadamard | 9.80749 | 0.00945901 | True |
| 9 | nar | 9.19644 | 0.00836444 | True |
| 9 | nar_rope | N/A | N/A | False |
| 10 | bf16 | 14.8896 | 0 | True |
| 10 | identity | 14.8896 | 0.0235674 | True |
| 10 | hadamard | 9.59201 | 0.00973557 | True |
| 10 | nar | 9.02393 | 0.00860607 | True |
| 10 | nar_rope | N/A | N/A | False |
| 11 | bf16 | 19.5933 | 0 | True |
| 11 | identity | 19.5933 | 0.0333365 | True |
| 11 | hadamard | 10.1897 | 0.00907562 | True |
| 11 | nar | 9.63915 | 0.0081479 | True |
| 11 | nar_rope | N/A | N/A | False |
| 12 | bf16 | 14.2261 | 0 | True |
| 12 | identity | 14.2261 | 0.0204013 | True |
| 12 | hadamard | 9.8261 | 0.00971567 | True |
| 12 | nar | 9.16893 | 0.00849985 | True |
| 12 | nar_rope | N/A | N/A | False |
| 13 | bf16 | 15.7171 | 0 | True |
| 13 | identity | 15.7171 | 0.0230265 | True |
| 13 | hadamard | 10.3635 | 0.00966417 | True |
| 13 | nar | 9.66759 | 0.00838971 | True |
| 13 | nar_rope | N/A | N/A | False |
| 14 | bf16 | 15.537 | 0 | True |
| 14 | identity | 15.537 | 0.0238015 | True |
| 14 | hadamard | 9.97737 | 0.00953148 | True |
| 14 | nar | 8.96868 | 0.00772313 | True |
| 14 | nar_rope | N/A | N/A | False |
| 15 | bf16 | 14.7322 | 0 | True |
| 15 | identity | 14.7322 | 0.0226827 | True |
| 15 | hadamard | 9.79018 | 0.00998359 | True |
| 15 | nar | 8.99961 | 0.00838879 | True |
| 15 | nar_rope | N/A | N/A | False |
| 16 | bf16 | 15.5123 | 0 | True |
| 16 | identity | 15.5123 | 0.0235765 | True |
| 16 | hadamard | 9.86971 | 0.00947015 | True |
| 16 | nar | 9.15208 | 0.00816563 | True |
| 16 | nar_rope | N/A | N/A | False |
| 17 | bf16 | 15.0868 | 0 | True |
| 17 | identity | 15.0868 | 0.0246196 | True |
| 17 | hadamard | 9.50998 | 0.00960588 | True |
| 17 | nar | 8.75909 | 0.00815665 | True |
| 17 | nar_rope | N/A | N/A | False |
| 18 | bf16 | 14.7516 | 0 | True |
| 18 | identity | 14.7516 | 0.02239 | True |
| 18 | hadamard | 9.72976 | 0.00970753 | True |
| 18 | nar | 8.92863 | 0.0082337 | True |
| 18 | nar_rope | N/A | N/A | False |
| 19 | bf16 | 16.2104 | 0 | True |
| 19 | identity | 16.2104 | 0.0255458 | True |
| 19 | hadamard | 9.90505 | 0.00939289 | True |
| 19 | nar | 9.24158 | 0.00819677 | True |
| 19 | nar_rope | N/A | N/A | False |
| 20 | bf16 | 16.1882 | 0 | True |
| 20 | identity | 16.1882 | 0.0269162 | True |
| 20 | hadamard | 9.75653 | 0.00965045 | True |
| 20 | nar | 8.8445 | 0.00796608 | True |
| 20 | nar_rope | N/A | N/A | False |
| 21 | bf16 | 16.4023 | 0 | True |
| 21 | identity | 16.4023 | 0.0288024 | True |
| 21 | hadamard | 9.44226 | 0.00932262 | True |
| 21 | nar | 8.92697 | 0.00836669 | True |
| 21 | nar_rope | N/A | N/A | False |
| 22 | bf16 | 17.354 | 0 | True |
| 22 | identity | 17.354 | 0.0330339 | True |
| 22 | hadamard | 9.31328 | 0.00911754 | True |
| 22 | nar | 8.48245 | 0.00757608 | True |
| 22 | nar_rope | N/A | N/A | False |
| 23 | bf16 | 16.3054 | 0 | True |
| 23 | identity | 16.3054 | 0.0273014 | True |
| 23 | hadamard | 9.71364 | 0.0094556 | True |
| 23 | nar | 9.09268 | 0.00829526 | True |
| 23 | nar_rope | N/A | N/A | False |
| 24 | bf16 | 16.0967 | 0 | True |
| 24 | identity | 16.0967 | 0.0267973 | True |
| 24 | hadamard | 9.49404 | 0.00913014 | True |
| 24 | nar | 8.99823 | 0.0082202 | True |
| 24 | nar_rope | N/A | N/A | False |
| 25 | bf16 | 18.0643 | 0 | True |
| 25 | identity | 18.0643 | 0.0330596 | True |
| 25 | hadamard | 9.51923 | 0.00890238 | True |
| 25 | nar | 8.53949 | 0.0071701 | True |
| 25 | nar_rope | N/A | N/A | False |
| 26 | bf16 | 14.6599 | 0 | True |
| 26 | identity | 14.6599 | 0.0240775 | True |
| 26 | hadamard | 9.27944 | 0.00953135 | True |
| 26 | nar | 8.58834 | 0.00816212 | True |
| 26 | nar_rope | N/A | N/A | False |
| 27 | bf16 | 16.7532 | 0 | True |
| 27 | identity | 16.7532 | 0.028945 | True |
| 27 | hadamard | 9.40494 | 0.00896148 | True |
| 27 | nar | 8.85237 | 0.00799134 | True |
| 27 | nar_rope | N/A | N/A | False |

## E2 — KV-only perplexity proxy

Only K and V are fake-quantized; weights and all other activations remain bf16. K is rotated after RoPE and Q receives the identical orthogonal rotation. The full-sequence prefill tensors are quantized at the same point they enter the cache; this is a cache-content proxy, not an autoregressive latency benchmark. Test sequences, calibration data, and all non-R settings are paired. Deterministic bf16/identity evaluations are measured once and reused exactly across seed rows.

| model | b | method | mean PPL | seed SD | paired delta vs Had | 90% CI low | 90% CI high | seed PPLs |
|---|---|---|---|---|---|---|---|---|
| llama32_3b | 64 | bf16 | 7.61655 | 0 | -0.119123 | -0.12998 | -0.108267 | 7.61655149;7.61655149;7.61655149 |
| llama32_3b | 64 | hadamard | 7.73567 | 0.00643994 | 0 | 0 | 0 | 7.74296257;7.733312;7.73075024 |
| llama32_3b | 64 | identity | 7.80385 | 0 | 0.0681739 | 0.0573171 | 0.0790307 | 7.80384882;7.80384882;7.80384882 |
| llama32_3b | 64 | nar | 7.70657 | 0.00088966 | -0.029102 | -0.0392892 | -0.0189148 | 7.70725479;7.70556656;7.70689737 |
| llama32_3b | 64 | nar_rope | 7.72 | 0.00600981 | -0.0156717 | -0.0360164 | 0.00467304 | 7.71336694;7.72507847;7.72156437 |
| llama32_3b | 128 | bf16 | 7.61655 | 0 | -0.165856 | -0.175301 | -0.15641 | 7.61655149;7.61655149;7.61655149 |
| llama32_3b | 128 | hadamard | 7.78241 | 0.00560303 | 0 | 0 | 0 | 7.78166119;7.78834558;7.77721422 |
| llama32_3b | 128 | identity | 8.00706 | 0 | 0.224652 | 0.215206 | 0.234098 | 8.00705928;8.00705928;8.00705928 |
| llama32_3b | 128 | nar | 7.76527 | 0.00109193 | -0.0171341 | -0.0278942 | -0.00637397 | 7.76443563;7.76487511;7.76650796 |
| llama32_1b | 32 | bf16 | 9.52734 | 0 | -0.374289 | -0.393114 | -0.355464 | 9.52734406;9.52734406;9.52734406 |
| llama32_1b | 32 | hadamard | 9.90163 | 0.0111664 | 0 | 0 | 0 | 9.90776974;9.90838613;9.8887445 |
| llama32_1b | 32 | identity | 10.0253 | 0 | 0.123619 | 0.104794 | 0.142444 | 10.0252528;10.0252528;10.0252528 |
| llama32_1b | 32 | nar | 9.78784 | 0.00647878 | -0.113791 | -0.123403 | -0.104178 | 9.7886786;9.79386273;9.78098635 |
| llama32_1b | 32 | nar_rope | 9.83579 | 0.00640406 | -0.0658458 | -0.0945095 | -0.0371821 | 9.83604074;9.82926079;9.8420614 |

## Negative findings and diagnosis

The primary phenomenon criterion failed because the required b=128 NAR reduction was only 7.770% (threshold 10%); only 5/28 layers individually reached 10%.
The spectrum was not flat (mean effective rank 23.110/128; top-1 trace share 12.201%), and disjoint calibration retained 100.914% of the A reduction, so flat spectrum and calibration overfit do not explain the failure.
Instead, b=128 has only one DC slot: the matched k=0 to k=1 ablation reduced range by 5.476%, despite 15.231% projection-removal attribution. The residual directions and the non-additivity of range dominate after absorbing one direction.
The group-size trend supports this capacity diagnosis: NAR reductions were 20.538% at b=32, 12.656% at b=64, and 7.770% at b=128.
BOS is not the driver in these samples: its identity range was only 0.218x non-BOS.
KV-only PPL nevertheless favored NAR, showing that the range threshold and the cache proxy can disagree; by the stated conjunctive criteria this is still a no-go.


## Unsure about

- The prompt does not define a numeric threshold for a 'substantial' top-k range share; 10% was fixed here and is reported separately so it can be reinterpreted without rerunning.
- fp16 storage of the real-valued zero/offset makes exact constant-shift MSE invariance mathematically impossible for arbitrary constants because the offset's fp16 rounding grid changes with magnitude. E0 therefore reports every measured delta and uses a fixed 2%-of-baseline-MSE tolerance; the range itself is invariant.
- NAR-RoPE is undefined when there is only one group, including 3B b=128 and 1B b=64. Reporting identity under that name would fabricate a method, so those cells are N/A.
- Three seeds support only a very low-degree-of-freedom CI. The paired 90% CI is the requested result, not strong evidence of distributional generality.

## Go / no-go

**NO-GO:** the core tensor premise does not meet the stated gate. Do not build a full quantization pipeline around this version of NAR.

## Reproduction artifacts

All exact numeric CSVs are under `results/`; captured samples and all-token second moments are under `activations/`; executed commands and stdout/stderr logs are under `runs/`. Run `./run_all.sh` to reproduce every stage and table.
