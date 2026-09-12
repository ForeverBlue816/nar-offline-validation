Random weights;128 causal steps, first8 discarded. Pooled run dispersion is not session dispersion.

| Model | Method | Mode | ms/token | Run std | Speedup | Sessions |
| --- | --- | --- | --- | --- | --- | --- |
| 3b | FP16 | eager_sequence | 21.02 | 0.24 | 1.00 | 1 |
| 3b | FP16 | eager_step_sync | 23.12 | 0.18 | 1.00 | 1 |
| 3b | QuaRot Hadamard | eager_sequence | 43.47 | 0.35 | 0.48 | 1 |
| 3b | QuaRot Hadamard | eager_step_sync | 43.10 | 0.45 | 0.54 | 1 |
| 3b | PrismQuant k=8 | eager_sequence | 41.27 | 0.34 | 0.51 | 1 |
| 3b | PrismQuant k=8 | eager_step_sync | N/A | N/A | N/A | 0 |
| 8b | FP16 | eager_sequence | N/A | N/A | N/A | 0 |
| 8b | FP16 | eager_step_sync | N/A | N/A | N/A | 0 |
| 8b | QuaRot Hadamard | eager_sequence | N/A | N/A | N/A | 0 |
| 8b | QuaRot Hadamard | eager_step_sync | N/A | N/A | N/A | 0 |
| 8b | PrismQuant k=8 | eager_sequence | N/A | N/A | N/A | 0 |
| 8b | PrismQuant k=8 | eager_step_sync | N/A | N/A | N/A | 0 |
