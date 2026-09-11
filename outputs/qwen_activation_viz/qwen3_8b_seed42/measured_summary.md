# Measured results

Qwen3-8B-Base; eight fixed WikiText-2 test windows; 2048 tokens/window; seed 42; frozen rotation seed 0. The following results are for paired local down_proj inputs. They use complete tensors and pooled energy ratios.

| Block | Method | Maximum absolute value | Mean group range | NMSE | Common-component energy (%) |
|---:|---|---:|---:|---:|---:|
| 1 | Hadamard | 0.48 | 0.13 | 7.94e-03 | 0.81 |
| 1 | PrismQuant (k=8) | 0.89 | 0.08 | 4.49e-03 | 7.21 |
| 1 | PrismQuant (k=max) | 0.88 | 0.08 | 4.19e-03 | 20.10 |
| 13 | Hadamard | 1.01 | 0.52 | 9.88e-03 | 0.75 |
| 13 | PrismQuant (k=8) | 1.91 | 0.45 | 7.64e-03 | 13.13 |
| 13 | PrismQuant (k=max) | 1.83 | 0.42 | 6.60e-03 | 25.54 |
| 24 | Hadamard | 2.51 | 1.57 | 9.61e-03 | 0.80 |
| 24 | PrismQuant (k=8) | 6.49 | 1.22 | 6.30e-03 | 9.44 |
| 24 | PrismQuant (k=max) | 6.87 | 1.15 | 5.59e-03 | 23.20 |
| 36 | Hadamard | 84.17 | 68.97 | 6.07e-03 | 0.05 |
| 36 | PrismQuant (k=8) | 361.33 | 6.53 | 6.44e-05 | 98.85 |
| 36 | PrismQuant (k=max) | 357.54 | 4.95 | 3.74e-05 | 99.44 |

Across these four preselected blocks, both PrismQuant settings have larger maximum absolute values than Hadamard and smaller mean group ranges and actual activation-QDQ NMSE. This directly shows why raw peak height and quantization quality need to be evaluated separately. Group centering is an analysis decomposition: it leaves the range of every token/group unchanged. It is not a deployed subtraction or the quantizer’s stored offset.

The strongest measured change is at Block 36. For k=max, common-component energy accounts for 99.44% of total activation energy. This is consistent with group alignment of dominant activation directions. It does not establish a downstream perplexity/accuracy improvement or attention-sink causality. The q_proj control and separately captured end-to-end rows are available in the complete CSV tables.

Required data/energy/complete-projection/capture checks pass. Three supplementary probes exceed their unchanged tolerances: one inverse check and two narrow-output checks on Block 36 frozen factors. Overall all-checks status remains FAIL; the initial failed probes and exact factor normalization deviations are retained in qa/. No checkpoint or factor was renormalized.

[All metrics](metrics_summary.csv) · [Per-sample variation](metrics_per_sample.csv) · [Validation](validation_report.json)
