# NAR offline tensor validation

A reproducible benchmark for normalized-anchor rotations (NAR) under dynamic
asymmetric INT4 quantization. It covers post-RoPE Llama K tensors, wide
`q_proj`/`down_proj` inputs, activation-only perplexity, factorized online
cost, per-token V-cache quantization, and a KIVI-style per-channel K baseline.

> **Corrected result:** the valid pre-registered K gates at `b=32/64` pass, as
> do all frozen E2 NAR rows. The original `b=128` K gate is invalid because a
> 128-dimensional head contains only one group and therefore one DC slot. In
> the fair K-axis comparison, however, per-channel K quantization beats every
> tested per-token rotation method.

| Check | Result | Main number |
|---|---:|---:|
| Corrected E1 K, `b=32` | PASS | 20.489% range reduction vs Hadamard |
| Corrected E1 K, `b=64` | PASS | 12.612% range reduction vs Hadamard |
| All frozen E2 NAR rows | PASS | Paired PPL deltas satisfy the frozen gate |
| E1b position robustness | PASS | 19.602–20.978% (`b=32`), 11.984–12.788% (`b=64`) |
| E1c `q_proj` input | positive | 23.755% mean paired-layer range reduction vs full Hadamard |
| E1c `down_proj` input | positive | 25.301% mean paired-layer range reduction vs full Hadamard |
| E1d KIVI-style baseline | **clear winner** | Lower NMSE in 28/28 layers at both group sizes |
| E5 activation PPL, 3B | positive | NAR beats Hadamard at qkv/down/both; paired 90% CIs exclude zero |
| E5 activation PPL, 1B | positive | NAR beats Hadamard at qkv/down/both; paired 90% CIs exclude zero |
| E5 activation PPL, 8B | positive | NAR beats Hadamard at qkv/down/both; paired 90% CIs exclude zero |
| E6 unfused online cost | **FAIL** | NAR wall time exceeds 10% of `down_proj` matmul at every measured token count |
| E7 per-token V cache | positive, modest | NAR reduces mean range and NMSE vs Hadamard at `b=32/64/128` |
| E8 range-direct refinement | negative for range | Held-out mean range worsens by 0.000560; NMSE improves by 0.000012 |

NAR-RoPE is dominated by plain NAR in every available paired range/NMSE check
and at every paired E2 seed, so it is dropped from further work. These are
paired, no-tuning results; negative findings and randomized-eigenspace residuals
are reported rather than filtered.

![E1c mean range versus absorbed rank](results/llama32_3b/e1c_range_vs_k.png)

![E1c realized range versus absorbed energy](results/llama32_3b/e1c_energy_fit.png)

## Scope and artifacts

The repository performs forward-hook activation capture, offline tensor
analysis, KV-only fake quantization, and activation-only perplexity proxies. It
does **not** run GPTQ, weight quantization, end-to-end W4A4KV4, or configuration
tuning. Models are Llama-3.2-3B, Llama-3.2-1B, and Llama-3.1-8B; data are fixed
WikiText-2 chunks.

The complete protocol, exact fit statistics, summary tables, confidence
intervals, caveats, and results are in [`report.md`](report.md). Exact per-layer
CSV/JSON outputs and figures are committed under [`results/`](results/), with execution
transcripts under [`runs/`](runs/). `results/decision_corrected.json` is the
authoritative decision; the older `results/decision.json` is retained only as
the archival outcome of the mis-specified `b=128` gate.

Large assets are intentionally excluded from Git: model weights, caches,
environments, and raw activations. The E1c capture retains exact bf16 bit
patterns for all 128×2048 tokens at every layer (168 GiB total) so that E3 FP4
E2M1 and E4 two-level NVFP4 can reuse it.

## Reproduce

The original frozen E0/E1/E2 pipeline is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

The E1b/E1c/E1d extension requires project storage because it writes the large
activation capture. Set `NAR_WORKDIR` to that storage and submit from the
repository root so Slurm can create relative log paths:

```bash
mkdir -p runs
NAR_WORKDIR=/path/to/project-storage sbatch slurm_extensions.sh
```

The activation continuation has separate jobs so models and diagnostics can
run independently:

```bash
NAR_WORKDIR=/path/to/project-storage sbatch slurm_activation_3b.sh
NAR_WORKDIR=/path/to/project-storage sbatch slurm_activation_1b.sh
NAR_WORKDIR=/path/to/project-storage sbatch slurm_activation_8b.sh
NAR_WORKDIR=/path/to/project-storage sbatch slurm_activation_diagnostics.sh
```

The batch script expects the environment at `$NAR_WORKDIR/venv`. See
[`nar/README.md`](nar/README.md) for frozen choices and individual stage
commands.
