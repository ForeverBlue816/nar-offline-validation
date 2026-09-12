Random weights;128 causal steps, first8 discarded. Pooled run dispersion is not session dispersion.

| Model | Method | Mode | ms/token | Run std | Speedup | Sessions |
| --- | --- | --- | --- | --- | --- | --- |
| 3b | FP16 | eager_sequence | 22.10 | 0.74 | 1.00 | 3 |
| 3b | FP16 | eager_step_sync | 23.19 | 0.27 | 1.00 | 3 |
| 3b | QuaRot Hadamard | eager_sequence | 43.51 | 0.51 | 0.51 | 3 |
| 3b | QuaRot Hadamard | eager_step_sync | 44.91 | 1.29 | 0.52 | 3 |
| 3b | PrismQuant k=8 | eager_sequence | 42.48 | 1.20 | 0.52 | 3 |
| 3b | PrismQuant k=8 | eager_step_sync | 44.78 | 1.08 | 0.52 | 3 |
| 8b | FP16 | eager_sequence | 30.21 | 0.03 | 1.00 | 3 |
| 8b | FP16 | eager_step_sync | 31.14 | 0.13 | 1.00 | 3 |
| 8b | QuaRot Hadamard | eager_sequence | 52.92 | 1.03 | 0.57 | 3 |
| 8b | QuaRot Hadamard | eager_step_sync | 55.20 | 1.70 | 0.56 | 3 |
| 8b | PrismQuant k=8 | eager_sequence | 50.61 | 0.63 | 0.60 | 3 |
| 8b | PrismQuant k=8 | eager_step_sync | 51.38 | 1.86 | 0.61 | 3 |
