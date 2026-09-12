Fixed layer0. FP16 module/stage/dispatch and E17 native packed-output contracts are distinct; Q is the shared QuaRot quantizer. Chains are directly timed. Complete records have three50-run sessions; invalid rows are not ranked.

| Model / T | Implementation | Wall us | Event us | Valid sessions | Status |
| --- | --- | --- | --- | --- | --- |
| 3b / 1 | NAR A | 36.94 | 32.77 | 3 | COMPLETE |
| 3b / 1 | NAR B (shuffle) | 30.09 | 30.72 | 3 | COMPLETE |
| 3b / 1 | NAR B (TC) | 33.81 | 35.84 | 3 | COMPLETE |
| 3b / 1 | Block-Had native | 29.42 | 30.72 | 3 | COMPLETE |
| 3b / 1 | Hadamard (FP16) | 68.03 | 70.66 | 3 | COMPLETE |
| 3b / 1 | Hadamard + Q | 152.80 | 151.04 | 3 | COMPLETE |
| 3b / 1 | NAR generic | 57.42 | 59.39 | 3 | COMPLETE |
| 3b / 1 | NAR shuffle | 52.38 | 53.25 | 3 | COMPLETE |
| 3b / 1 | NAR module | 44.62 | 46.61 | 3 | COMPLETE |
| 3b / 1 | NAR native | 70.88 | 72.70 | 3 | COMPLETE |
| 3b / 1 | NAR + Q | 121.23 | 122.88 | 3 | COMPLETE |
| 3b / 1 | NAR prebound | 31.09 | 31.74 | 3 | COMPLETE |
| 3b / 1 | Shared quantizer | 74.18 | 75.78 | 3 | COMPLETE |
| 3b / 2048 | NAR A | 83.23 | 78.85 | 3 | COMPLETE |
| 3b / 2048 | NAR B (shuffle) | 507.61 | 502.78 | 3 | COMPLETE |
| 3b / 2048 | NAR B (TC) | 155.91 | 150.53 | 3 | COMPLETE |
| 3b / 2048 | Block-Had native | 453.36 | 447.49 | 3 | COMPLETE |
| 3b / 2048 | Hadamard (FP16) | 169.10 | 161.79 | 3 | COMPLETE |
| 3b / 2048 | Hadamard + Q | 423.37 | 412.67 | 3 | COMPLETE |
| 3b / 2048 | NAR generic | 216.62 | 211.97 | 3 | COMPLETE |
| 3b / 2048 | NAR shuffle | 570.53 | 565.25 | 3 | COMPLETE |
| 3b / 2048 | NAR module | 212.82 | 207.86 | 3 | COMPLETE |
| 3b / 2048 | NAR native | 580.62 | 575.44 | 3 | COMPLETE |
| 3b / 2048 | NAR + Q | 456.04 | 452.58 | 3 | COMPLETE |
| 3b / 2048 | NAR prebound | 199.15 | 195.58 | 3 | COMPLETE |
| 3b / 2048 | Shared quantizer | 260.49 | 254.98 | 3 | COMPLETE |
| 3b / 32768 | NAR A | 886.11 | 879.62 | 3 | COMPLETE |
| 3b / 32768 | NAR B (shuffle) | 6090.75 | 6082.56 | 3 | COMPLETE |
| 3b / 32768 | NAR B (TC) | 1818.79 | 1811.46 | 3 | COMPLETE |
| 3b / 32768 | Block-Had native | 6758.23 | 6749.18 | 3 | COMPLETE |
| 3b / 32768 | Hadamard (FP16) | 1737.44 | 1727.49 | 3 | COMPLETE |
| 3b / 32768 | Hadamard + Q | 5347.88 | 5331.46 | 3 | COMPLETE |
| 3b / 32768 | NAR generic | 2677.09 | 2668.54 | 3 | COMPLETE |
| 3b / 32768 | NAR shuffle | 6946.76 | 6938.62 | 3 | COMPLETE |
| 3b / 32768 | NAR module | 2677.44 | 2665.47 | 3 | COMPLETE |
| 3b / 32768 | NAR native | 11982.37 | 11973.63 | 3 | COMPLETE |
| 3b / 32768 | NAR + Q | 6260.16 | 6248.45 | 3 | COMPLETE |
| 3b / 32768 | NAR prebound | 2649.33 | 2640.90 | 3 | COMPLETE |
| 3b / 32768 | Shared quantizer | 3592.55 | 3582.98 | 3 | COMPLETE |
| 8b / 1 | NAR A | 42.74 | 38.91 | 3 | COMPLETE |
| 8b / 1 | NAR B (shuffle) | 32.37 | 30.72 | 3 | COMPLETE |
| 8b / 1 | NAR B (TC) | 36.07 | 36.86 | 3 | COMPLETE |
| 8b / 1 | Block-Had native | 30.41 | 30.72 | 3 | COMPLETE |
| 8b / 1 | Hadamard (FP16) | 123.67 | 123.87 | 3 | COMPLETE |
| 8b / 1 | Hadamard + Q | 217.45 | 219.14 | 3 | COMPLETE |
| 8b / 1 | NAR generic | 58.45 | 59.39 | 3 | COMPLETE |
| 8b / 1 | NAR shuffle | 55.19 | 54.27 | 3 | COMPLETE |
| 8b / 1 | NAR module | 47.69 | 50.69 | 3 | COMPLETE |
| 8b / 1 | NAR native | 57.86 | 55.30 | 3 | COMPLETE |
| 8b / 1 | NAR + Q | 123.04 | 125.44 | 3 | COMPLETE |
| 8b / 1 | NAR prebound | 36.81 | 33.79 | 3 | COMPLETE |
| 8b / 1 | Shared quantizer | 74.01 | 75.78 | 3 | COMPLETE |
| 8b / 2048 | NAR A | 121.12 | 116.74 | 3 | COMPLETE |
| 8b / 2048 | NAR B (shuffle) | 870.66 | 863.70 | 3 | COMPLETE |
| 8b / 2048 | NAR B (TC) | 231.05 | 225.28 | 3 | COMPLETE |
| 8b / 2048 | Block-Had native | 774.38 | 765.95 | 3 | COMPLETE |
| 8b / 2048 | Hadamard (FP16) | 445.07 | 435.71 | 3 | COMPLETE |
| 8b / 2048 | Hadamard + Q | 858.97 | 848.90 | 3 | COMPLETE |
| 8b / 2048 | NAR generic | 321.09 | 316.42 | 3 | COMPLETE |
| 8b / 2048 | NAR shuffle | 972.83 | 966.64 | 3 | COMPLETE |
| 8b / 2048 | NAR module | 321.30 | 315.87 | 3 | COMPLETE |
| 8b / 2048 | NAR native | 984.41 | 976.90 | 3 | COMPLETE |
| 8b / 2048 | NAR + Q | 735.99 | 729.57 | 3 | COMPLETE |
| 8b / 2048 | NAR prebound | 307.55 | 303.10 | 3 | COMPLETE |
| 8b / 2048 | Shared quantizer | 428.34 | 422.91 | 3 | COMPLETE |
| 8b / 32768 | NAR A | 1505.10 | 1498.11 | 3 | COMPLETE |
| 8b / 32768 | NAR B (shuffle) | 10634.67 | 10625.02 | 3 | COMPLETE |
| 8b / 32768 | NAR B (TC) | 3125.56 | 3119.10 | 3 | COMPLETE |
| 8b / 32768 | Block-Had native | 11763.69 | 11754.48 | 3 | COMPLETE |
| 8b / 32768 | Hadamard (FP16) | 5974.46 | 5956.61 | 3 | COMPLETE |
| 8b / 32768 | Hadamard + Q | 12174.74 | 12148.74 | 3 | COMPLETE |
| 8b / 32768 | NAR generic | 4587.56 | 4579.84 | 3 | COMPLETE |
| 8b / 32768 | NAR shuffle | 12063.54 | 12058.11 | 3 | COMPLETE |
| 8b / 32768 | NAR module | 4606.58 | 4586.50 | 3 | COMPLETE |
| 8b / 32768 | NAR native | 20906.34 | 20895.23 | 3 | COMPLETE |
| 8b / 32768 | NAR + Q | 10825.50 | 10802.18 | 3 | COMPLETE |
| 8b / 32768 | NAR prebound | 4581.87 | 4568.06 | 3 | COMPLETE |
| 8b / 32768 | Shared quantizer | 6234.65 | 6224.38 | 3 | COMPLETE |
