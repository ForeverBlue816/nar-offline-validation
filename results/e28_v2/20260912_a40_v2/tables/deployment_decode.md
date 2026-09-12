Random weights;128 causal steps, first8 discarded. Pooled run dispersion is not session dispersion.

| Model | Method | Mode | ms/token | Run std | Speedup | Sessions |
| --- | --- | --- | --- | --- | --- | --- |
| 3b | FP16 | eager_sequence | N/A | N/A | N/A | 0 |
| 3b | FP16 | eager_step_sync | N/A | N/A | N/A | 0 |
| 3b | QuaRot Hadamard | eager_sequence | N/A | N/A | N/A | 0 |
| 3b | QuaRot Hadamard | eager_step_sync | N/A | N/A | N/A | 0 |
| 3b | PrismQuant k=8 | eager_sequence | N/A | N/A | N/A | 0 |
| 3b | PrismQuant k=8 | eager_step_sync | N/A | N/A | N/A | 0 |
| 8b | FP16 | eager_sequence | N/A | N/A | N/A | 0 |
| 8b | FP16 | eager_step_sync | N/A | N/A | N/A | 0 |
| 8b | QuaRot Hadamard | eager_sequence | N/A | N/A | N/A | 0 |
| 8b | QuaRot Hadamard | eager_step_sync | N/A | N/A | N/A | 0 |
| 8b | PrismQuant k=8 | eager_sequence | N/A | N/A | N/A | 0 |
| 8b | PrismQuant k=8 | eager_step_sync | N/A | N/A | N/A | 0 |
