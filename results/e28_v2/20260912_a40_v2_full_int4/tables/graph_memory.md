Private panel; independent-process inference peaks. Capture preparation is separate from steady-state timing. Net capture allocated/reserved changes do not isolate private-pool size; negative reserved changes are retained in JSON. Eager capture fields are N/A.

| Model | Method | Mode | Peak alloc GB | Peak reserv GB | Graphs | Capture s | Net alloc MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3b | fp16 | cuda_graph_sequence | 8.74 | 8.85 | 1 | 0.58 | 9.03 |
| 3b | fp16 | eager_sequence | 8.72 | 8.80 | N/A | N/A | N/A |
| 3b | hadamard | cuda_graph_sequence | 4.37 | 4.52 | 1 | 0.58 | 9.03 |
| 3b | hadamard | eager_sequence | 4.35 | 4.47 | N/A | N/A | N/A |
| 3b | nar | cuda_graph_sequence | 4.39 | 4.54 | 1 | 0.53 | 9.03 |
| 3b | nar | eager_sequence | 4.37 | 4.49 | N/A | N/A | N/A |
| 8b | fp16 | cuda_graph_sequence | 18.77 | 19.03 | 1 | 0.94 | 9.03 |
| 8b | fp16 | eager_sequence | 18.75 | 18.99 | N/A | N/A | N/A |
| 8b | hadamard | cuda_graph_sequence | 8.15 | 8.33 | 1 | 0.87 | 9.03 |
| 8b | hadamard | eager_sequence | 8.13 | 8.28 | N/A | N/A | N/A |
| 8b | nar | cuda_graph_sequence | 8.19 | 8.40 | 1 | 0.78 | 9.03 |
| 8b | nar | eager_sequence | 8.18 | 8.35 | N/A | N/A | N/A |
