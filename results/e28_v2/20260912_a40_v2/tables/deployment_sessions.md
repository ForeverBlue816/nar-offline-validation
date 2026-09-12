Core session medians and run population std. Prefill units are total input tok/s; decode units are ms/causal step (batch8 advances eight sequences per step). Session-median dispersion is separate in metrics_summary.json.

| Model | Method | Mode / phase | Session | Median | Run std | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| 3b | FP16 | eager_sequence / prefill1 | 1 | 10116.33 | 18.19 | 50 |
| 3b | FP16 | eager_sequence / prefill16 | 1 | 13973.38 | 58.48 | 50 |
