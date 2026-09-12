Performance, numerical consistency and native-paper-format deployment are separate claims.

| Path | Format | Weights | Numerical status | Deployment status |
| --- | --- | --- | --- | --- |
| E28 kernel swap | Signed INT4 W/A; token-channel affine KV4 | Random weights | See correctness JSON | Paper format not connected |
| E17 native R4 | Group128 asymmetric packed A4 | Real calibrated factors | Local only | No model GEMM/KV claim |
| Paper accuracy path | GPTQ W4; group128 asymmetric A4; K/V residual32 | Local k8 checkpoint absent | BLOCKED | Current row/column-scale GEMM incompatible |
