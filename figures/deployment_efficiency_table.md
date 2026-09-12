# Frozen E28 plotting values

Displayed values use two decimals; the CSV retains unrounded values.

| Metric | Model | Method | Baseline | Mode / scope | B / T | Value | Unit |
|---|---|---|---|---|---|---:|---|
| prefill_throughput | 3b | fp16 |  | eager_sequence / prefill1 | 1 / 2048 | 10163.48 | input tokens/s |
| prefill_throughput | 3b | hadamard |  | eager_sequence / prefill1 | 1 / 2048 | 12819.74 | input tokens/s |
| prefill_speedup | 3b | hadamard | fp16 | eager_sequence / prefill1 | 1 / 2048 | 1.26 | ratio |
| prefill_throughput | 3b | nar |  | eager_sequence / prefill1 | 1 / 2048 | 12674.40 | input tokens/s |
| prefill_speedup | 3b | nar | fp16 | eager_sequence / prefill1 | 1 / 2048 | 1.25 | ratio |
| prefill_throughput | 3b | fp16 |  | eager_sequence / prefill16 | 16 / 2048 | 13985.13 | input tokens/s |
| prefill_throughput | 3b | hadamard |  | eager_sequence / prefill16 | 16 / 2048 | 17894.66 | input tokens/s |
| prefill_speedup | 3b | hadamard | fp16 | eager_sequence / prefill16 | 16 / 2048 | 1.28 | ratio |
| prefill_throughput | 3b | nar |  | eager_sequence / prefill16 | 16 / 2048 | 17630.27 | input tokens/s |
| prefill_speedup | 3b | nar | fp16 | eager_sequence / prefill16 | 16 / 2048 | 1.26 | ratio |
| prefill_throughput | 8b | fp16 |  | eager_sequence / prefill1 | 1 / 2048 | 5241.70 | input tokens/s |
| prefill_throughput | 8b | hadamard |  | eager_sequence / prefill1 | 1 / 2048 | 7607.40 | input tokens/s |
| prefill_speedup | 8b | hadamard | fp16 | eager_sequence / prefill1 | 1 / 2048 | 1.45 | ratio |
| prefill_throughput | 8b | nar |  | eager_sequence / prefill1 | 1 / 2048 | 7677.01 | input tokens/s |
| prefill_speedup | 8b | nar | fp16 | eager_sequence / prefill1 | 1 / 2048 | 1.46 | ratio |
| prefill_throughput | 8b | fp16 |  | eager_sequence / prefill16 | 16 / 2048 | 6418.43 | input tokens/s |
| prefill_throughput | 8b | hadamard |  | eager_sequence / prefill16 | 16 / 2048 | 9556.00 | input tokens/s |
| prefill_speedup | 8b | hadamard | fp16 | eager_sequence / prefill16 | 16 / 2048 | 1.49 | ratio |
| prefill_throughput | 8b | nar |  | eager_sequence / prefill16 | 16 / 2048 | 9661.75 | input tokens/s |
| prefill_speedup | 8b | nar | fp16 | eager_sequence / prefill16 | 16 / 2048 | 1.51 | ratio |
| decode_latency | 3b | fp16 |  | eager_sequence / decode | 1 / 1 | 21.86 | ms/token |
| decode_speedup | 3b | fp16 | fp16 | eager_sequence / decode | 1 / 1 | 1.00 | ratio |
| decode_latency | 3b | hadamard |  | eager_sequence / decode | 1 / 1 | 42.83 | ms/token |
| decode_speedup | 3b | hadamard | fp16 | eager_sequence / decode | 1 / 1 | 0.51 | ratio |
| decode_latency | 3b | nar |  | eager_sequence / decode | 1 / 1 | 43.08 | ms/token |
| decode_speedup | 3b | nar | fp16 | eager_sequence / decode | 1 / 1 | 0.51 | ratio |
| decode_overhead | 3b | nar | hadamard | eager_sequence / decode | 1 / 1 | 0.60 | % |
| peak_memory | 3b | fp16 |  | eager_sequence / decode | 1 / 1 | 8.72 | GB |
| peak_memory | 3b | hadamard |  | eager_sequence / decode | 1 / 1 | 4.35 | GB |
| peak_memory | 3b | nar |  | eager_sequence / decode | 1 / 1 | 4.37 | GB |
| memory_saving | 3b | hadamard | fp16 | eager_sequence / decode | 1 / 1 | 50.07 | % |
| memory_saving | 3b | nar | fp16 | eager_sequence / decode | 1 / 1 | 49.82 | % |
| decode_latency | 3b | fp16 |  | cuda_graph_sequence / decode | 1 / 1 | 15.04 | ms/token |
| decode_speedup | 3b | fp16 | fp16 | cuda_graph_sequence / decode | 1 / 1 | 1.00 | ratio |
| decode_latency | 3b | hadamard |  | cuda_graph_sequence / decode | 1 / 1 | 16.33 | ms/token |
| decode_speedup | 3b | hadamard | fp16 | cuda_graph_sequence / decode | 1 / 1 | 0.92 | ratio |
| decode_latency | 3b | nar |  | cuda_graph_sequence / decode | 1 / 1 | 16.65 | ms/token |
| decode_speedup | 3b | nar | fp16 | cuda_graph_sequence / decode | 1 / 1 | 0.90 | ratio |
| decode_overhead | 3b | nar | hadamard | cuda_graph_sequence / decode | 1 / 1 | 1.96 | % |
| peak_memory | 3b | fp16 |  | cuda_graph_sequence / decode | 1 / 1 | 8.74 | GB |
| peak_memory | 3b | hadamard |  | cuda_graph_sequence / decode | 1 / 1 | 4.37 | GB |
| peak_memory | 3b | nar |  | cuda_graph_sequence / decode | 1 / 1 | 4.39 | GB |
| memory_saving | 3b | hadamard | fp16 | cuda_graph_sequence / decode | 1 / 1 | 49.97 | % |
| memory_saving | 3b | nar | fp16 | cuda_graph_sequence / decode | 1 / 1 | 49.72 | % |
| graph_vs_eager | 3b | fp16 | fp16 | matched eager / graph / decode | 1 / 1 | 1.45 | ratio |
| graph_vs_eager | 3b | hadamard | hadamard | matched eager / graph / decode | 1 / 1 | 2.62 | ratio |
| graph_vs_eager | 3b | nar | nar | matched eager / graph / decode | 1 / 1 | 2.59 | ratio |
| decode_latency | 8b | fp16 |  | eager_sequence / decode | 1 / 1 | 30.22 | ms/token |
| decode_speedup | 8b | fp16 | fp16 | eager_sequence / decode | 1 / 1 | 1.00 | ratio |
| decode_latency | 8b | hadamard |  | eager_sequence / decode | 1 / 1 | 52.31 | ms/token |
| decode_speedup | 8b | hadamard | fp16 | eager_sequence / decode | 1 / 1 | 0.58 | ratio |
| decode_latency | 8b | nar |  | eager_sequence / decode | 1 / 1 | 49.99 | ms/token |
| decode_speedup | 8b | nar | fp16 | eager_sequence / decode | 1 / 1 | 0.60 | ratio |
| decode_overhead | 8b | nar | hadamard | eager_sequence / decode | 1 / 1 | -4.43 | % |
| peak_memory | 8b | fp16 |  | eager_sequence / decode | 1 / 1 | 18.75 | GB |
| peak_memory | 8b | hadamard |  | eager_sequence / decode | 1 / 1 | 8.13 | GB |
| peak_memory | 8b | nar |  | eager_sequence / decode | 1 / 1 | 8.18 | GB |
| memory_saving | 8b | hadamard | fp16 | eager_sequence / decode | 1 / 1 | 56.63 | % |
| memory_saving | 8b | nar | fp16 | eager_sequence / decode | 1 / 1 | 56.39 | % |
| decode_latency | 8b | fp16 |  | cuda_graph_sequence / decode | 1 / 1 | 28.83 | ms/token |
| decode_speedup | 8b | fp16 | fp16 | cuda_graph_sequence / decode | 1 / 1 | 1.00 | ratio |
| decode_latency | 8b | hadamard |  | cuda_graph_sequence / decode | 1 / 1 | 23.10 | ms/token |
| decode_speedup | 8b | hadamard | fp16 | cuda_graph_sequence / decode | 1 / 1 | 1.25 | ratio |
| decode_latency | 8b | nar |  | cuda_graph_sequence / decode | 1 / 1 | 23.64 | ms/token |
| decode_speedup | 8b | nar | fp16 | cuda_graph_sequence / decode | 1 / 1 | 1.22 | ratio |
| decode_overhead | 8b | nar | hadamard | cuda_graph_sequence / decode | 1 / 1 | 2.35 | % |
| peak_memory | 8b | fp16 |  | cuda_graph_sequence / decode | 1 / 1 | 18.77 | GB |
| peak_memory | 8b | hadamard |  | cuda_graph_sequence / decode | 1 / 1 | 8.15 | GB |
| peak_memory | 8b | nar |  | cuda_graph_sequence / decode | 1 / 1 | 8.19 | GB |
| memory_saving | 8b | hadamard | fp16 | cuda_graph_sequence / decode | 1 / 1 | 56.58 | % |
| memory_saving | 8b | nar | fp16 | cuda_graph_sequence / decode | 1 / 1 | 56.34 | % |
| graph_vs_eager | 8b | fp16 | fp16 | matched eager / graph / decode | 1 / 1 | 1.05 | ratio |
| graph_vs_eager | 8b | hadamard | hadamard | matched eager / graph / decode | 1 / 1 | 2.26 | ratio |
| graph_vs_eager | 8b | nar | nar | matched eager / graph / decode | 1 / 1 | 2.11 | ratio |
| kernel_ratio | 3b | nar_module | hadamard_fp16 | wall time / E28 slot | 1 / 1 | 0.75 | ratio |
| kernel_ratio | 3b | nar_plus_quantizer | hadamard_plus_quantizer | wall time / E28 frontend | 1 / 1 | 0.80 | ratio |
| kernel_ratio | 3b | nar_prebound | nar_generic | wall time / Dispatch | 1 / 1 | 0.53 | ratio |
| kernel_ratio | 3b | nar_generic | nar_generic_shuffle | wall time / B implementation | 1 / 1 | 1.10 | ratio |
| kernel_ratio | 3b | nar_native | block_hadamard_native | wall time / E17 native | 1 / 1 | 2.59 | ratio |
| kernel_ratio | 3b | nar_module | hadamard_fp16 | wall time / E28 slot | 1 / 2048 | 1.30 | ratio |
| kernel_ratio | 3b | nar_plus_quantizer | hadamard_plus_quantizer | wall time / E28 frontend | 1 / 2048 | 1.08 | ratio |
| kernel_ratio | 3b | nar_prebound | nar_generic | wall time / Dispatch | 1 / 2048 | 0.91 | ratio |
| kernel_ratio | 3b | nar_generic | nar_generic_shuffle | wall time / B implementation | 1 / 2048 | 0.38 | ratio |
| kernel_ratio | 3b | nar_native | block_hadamard_native | wall time / E17 native | 1 / 2048 | 1.29 | ratio |
| kernel_ratio | 3b | nar_module | hadamard_fp16 | wall time / E28 slot | 1 / 32768 | 1.54 | ratio |
| kernel_ratio | 3b | nar_plus_quantizer | hadamard_plus_quantizer | wall time / E28 frontend | 1 / 32768 | 1.17 | ratio |
| kernel_ratio | 3b | nar_prebound | nar_generic | wall time / Dispatch | 1 / 32768 | 0.99 | ratio |
| kernel_ratio | 3b | nar_generic | nar_generic_shuffle | wall time / B implementation | 1 / 32768 | 0.39 | ratio |
| kernel_ratio | 3b | nar_native | block_hadamard_native | wall time / E17 native | 1 / 32768 | 1.77 | ratio |
| kernel_ratio | 8b | nar_module | hadamard_fp16 | wall time / E28 slot | 1 / 1 | 0.38 | ratio |
| kernel_ratio | 8b | nar_plus_quantizer | hadamard_plus_quantizer | wall time / E28 frontend | 1 / 1 | 0.58 | ratio |
| kernel_ratio | 8b | nar_prebound | nar_generic | wall time / Dispatch | 1 / 1 | 0.63 | ratio |
| kernel_ratio | 8b | nar_generic | nar_generic_shuffle | wall time / B implementation | 1 / 1 | 1.05 | ratio |
| kernel_ratio | 8b | nar_native | block_hadamard_native | wall time / E17 native | 1 / 1 | 1.92 | ratio |
| kernel_ratio | 8b | nar_module | hadamard_fp16 | wall time / E28 slot | 1 / 2048 | 0.73 | ratio |
| kernel_ratio | 8b | nar_plus_quantizer | hadamard_plus_quantizer | wall time / E28 frontend | 1 / 2048 | 0.85 | ratio |
| kernel_ratio | 8b | nar_prebound | nar_generic | wall time / Dispatch | 1 / 2048 | 0.96 | ratio |
| kernel_ratio | 8b | nar_generic | nar_generic_shuffle | wall time / B implementation | 1 / 2048 | 0.33 | ratio |
| kernel_ratio | 8b | nar_native | block_hadamard_native | wall time / E17 native | 1 / 2048 | 1.27 | ratio |
| kernel_ratio | 8b | nar_module | hadamard_fp16 | wall time / E28 slot | 1 / 32768 | 0.77 | ratio |
| kernel_ratio | 8b | nar_plus_quantizer | hadamard_plus_quantizer | wall time / E28 frontend | 1 / 32768 | 0.89 | ratio |
| kernel_ratio | 8b | nar_prebound | nar_generic | wall time / Dispatch | 1 / 32768 | 1.00 | ratio |
| kernel_ratio | 8b | nar_generic | nar_generic_shuffle | wall time / B implementation | 1 / 32768 | 0.38 | ratio |
| kernel_ratio | 8b | nar_native | block_hadamard_native | wall time / E17 native | 1 / 32768 | 1.78 | ratio |
