# E28-v2 frozen scope

P0 first: audit, numerical checks, three-session eager timing, independent memory attribution. P1: bounded graph feasibility and two fixed kernel ablations. P2: only decode b8/p2048 and b1/p8192, plus real-checkpoint interface audit.

All rows are random-weight performance experiments. A cached real-text input does not turn random weights into a quality evaluation. No full PPL/task evaluation is authorized by this plan.

The old reports/results are immutable. The v2 comparison uses common corrected RoPE caches, common shared KV metadata, and explicitly identical shared INT4 tensors. Therefore old and new times are not interchangeable. A separate legacy-wrapper diagnostic retains old synchronization behavior.

Thresholds, selected configurations, counts, exclusions and formulas were saved before formal timing in protocol.json. Pressure failures are retained without relaxing thresholds. No throughput target controls selection.
