Core session medians and run population std. Prefill units are total input tok/s; decode units are ms/causal step (batch8 advances eight sequences per step). Session-median dispersion is separate in metrics_summary.json.

| Model | Method | Mode / phase | Session | Median | Run std | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| 3b | FP16 | eager_sequence / decode | 1 | 21.02 | 0.24 | 50 |
| 3b | FP16 | eager_sequence / prefill1 | 1 | 10172.58 | 11.60 | 50 |
| 3b | FP16 | eager_sequence / prefill16 | 1 | 13915.92 | 69.35 | 50 |
| 3b | FP16 | eager_step_sync / decode | 1 | 23.12 | 0.18 | 50 |
| 3b | QuaRot Hadamard | eager_sequence / decode | 1 | 43.47 | 0.35 | 50 |
| 3b | QuaRot Hadamard | eager_sequence / prefill1 | 1 | 12824.19 | 28.34 | 50 |
| 3b | QuaRot Hadamard | eager_sequence / prefill16 | 1 | 17894.86 | 13.30 | 50 |
| 3b | QuaRot Hadamard | eager_step_sync / decode | 1 | 43.10 | 0.45 | 50 |
| 3b | PrismQuant k=8 | eager_sequence / decode | 1 | 41.27 | 0.34 | 50 |
| 3b | PrismQuant k=8 | eager_sequence / prefill1 | 1 | 12677.75 | 23.05 | 50 |
| 3b | PrismQuant k=8 | eager_sequence / prefill16 | 1 | 17634.42 | 19.59 | 50 |
