Independent process peak allocated bytes, GB=1e9. Reserved and NVML values are reported separately in JSON.

| Model | Method | Phase | Peak GB | Extra vs Had MB | Saved vs FP16 % |
| --- | --- | --- | --- | --- | --- |
| 3b | FP16 | prefill16 | 14.55 | N/A | 0.00 |
| 3b | FP16 | decode | 8.72 | N/A | 0.00 |
| 3b | QuaRot Hadamard | prefill16 | 8.10 | N/A | 44.32 |
| 3b | QuaRot Hadamard | decode | 4.35 | N/A | 50.07 |
| 3b | PrismQuant k=8 | prefill16 | 8.12 | 23.10 | 44.16 |
| 3b | PrismQuant k=8 | decode | 4.37 | 22.18 | 49.82 |
| 8b | FP16 | prefill16 | 26.47 | N/A | 0.00 |
| 8b | FP16 | decode | 18.75 | N/A | 0.00 |
| 8b | QuaRot Hadamard | prefill16 | 13.79 | N/A | 47.91 |
| 8b | QuaRot Hadamard | decode | 8.13 | N/A | 56.63 |
| 8b | PrismQuant k=8 | prefill16 | 13.83 | 45.06 | 47.74 |
| 8b | PrismQuant k=8 | decode | 8.18 | 44.14 | 56.39 |
