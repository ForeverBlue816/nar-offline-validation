Random weights; total input throughput. Core values require three sessions; incomplete rows are provisional. Run/session dispersion is in deployment_sessions.md.

| Model | Method | B1 tok/s | Speedup | B16 tok/s | Speedup |
| --- | --- | --- | --- | --- | --- |
| 3b | FP16 | 10163.48 | 1.00 | 13985.13 | 1.00 |
| 3b | QuaRot Hadamard | 12819.74 | 1.26 | 17894.66 | 1.28 |
| 3b | PrismQuant k=8 | 12674.40 | 1.25 | 17630.27 | 1.26 |
| 8b | FP16 | 5241.70 | 1.00 | 6418.43 | 1.00 |
| 8b | QuaRot Hadamard | 7607.40 | 1.45 | 9556.00 | 1.49 |
| 8b | PrismQuant k=8 | 7677.01 | 1.46 | 9661.75 | 1.51 |
