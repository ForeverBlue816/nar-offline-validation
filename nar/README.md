# NAR offline validation

`experiment.py` reproduces the frozen E0, E1, and E2 tables. The extension
`extended_experiment.py` reads those outputs without rerunning them, performs
E1b, captures the wide activations once, and evaluates E1c/E1d.
`activation_experiments.py` adds activation-only perplexity and factorized
online-cost checks (E5/E6). `activation_diagnostics.py` adds the per-token V
cache and one-shot range-direct diagnostics (E7/E8), while
`activation_report.py` appends their tables and figures to the corrected
report.

The experiment is limited to forward hooks, offline tensors, and KV-only fake
quantization. It never modifies weights, runs GPTQ, or builds a W4A4 pipeline.
All comparisons are paired and no result-driven configuration tuning is used.

## Frozen choices

- Llama weights: public exact-weight mirrors `unsloth/Llama-3.2-3B` and
  `unsloth/Llama-3.2-1B`; no fine-tuned model is used.
- E1 calibration A: first 128 BOS-prefixed, length-2048 WikiText-2 train
  chunks. Calibration B: the next disjoint 128 chunks.
- E2: first 64 BOS-prefixed, length-2048 WikiText-2 test chunks, three rotation
  seeds, paired methods.
- Corrected K gates: `b=32/64`. The original `b=128` K gate is archival only:
  `head_dim=128` gives one group and one DC slot.
- E1b probes exact stored K samples after model RoPE at positions
  `0/512/1024/2048`, using plain NAR only. NAR-RoPE is dropped as dominated.
- E1c captures every token at every layer for `q_proj` input
  (`n=3072`, 24 DC slots) and `down_proj` input (`n=8192`, 64 DC slots), with
  `b=128` and `k=0..n/b`.
- Top directions use all 262144 tokens with a fixed randomized symmetric
  eigensolver (oversampling 16, one power iteration, three passes). Range/NMSE
  and residual balancing use positions `0,32,...,2016` from all 128 sequences.
- E1d compares identical full post-RoPE K tensors against KIVI-style dynamic
  asymmetric per-channel INT4, grouping contiguous tokens at `b=32/64`.
- E5 uses group-128 dynamic asymmetric per-token INT4 at q/k/v and down_proj
  inputs only. Rotation seeds, evaluation chunks, and three sites are paired;
  all weights and non-target tensors remain bf16.
- E6 uses 64 Householder reflections followed by the fixed permutation, signs,
  and block H128, with dense equivalence checked only for validation.
- E7 rotates V independently within each KV head and folds the inverse into
  o_proj. Group sizes are fixed at `32/64/128`.
- E8 performs one seed and exactly 200 Grassmann steps with p=8 on calibration
  A, then evaluates the unchanged result on disjoint calibration B. There is no
  hyperparameter iteration.

## Commands

Original frozen pipeline:

```bash
python nar/experiment.py --workdir "$NAR_WORKDIR" e0
python nar/experiment.py --workdir "$NAR_WORKDIR" collect --model llama32_3b --tag cal_a
python nar/experiment.py --workdir "$NAR_WORKDIR" analyze --model llama32_3b
python nar/experiment.py --workdir "$NAR_WORKDIR" e2 --model llama32_3b
python nar/experiment.py --workdir "$NAR_WORKDIR" report
```

Extension stages (do not rerun completed frozen stages):

```bash
python nar/extended_experiment.py --workdir "$NAR_WORKDIR" e1b
python nar/extended_experiment.py --workdir "$NAR_WORKDIR" collect-wide --tag wide_cal_a
python nar/extended_experiment.py --workdir "$NAR_WORKDIR" e1c
python nar/extended_experiment.py --workdir "$NAR_WORKDIR" e1d
python nar/extended_experiment.py --workdir "$NAR_WORKDIR" report
```

Activation continuation stages:

```bash
python nar/activation_experiments.py --workdir "$NAR_WORKDIR" calibrate --model llama32_3b
python nar/activation_experiments.py --workdir "$NAR_WORKDIR" e5 --model llama32_3b
python nar/activation_experiments.py --workdir "$NAR_WORKDIR" e6
python nar/activation_diagnostics.py --workdir "$NAR_WORKDIR" collect-v
python nar/activation_diagnostics.py --workdir "$NAR_WORKDIR" e7
python nar/activation_diagnostics.py --workdir "$NAR_WORKDIR" collect-down-heldout
python nar/activation_diagnostics.py --workdir "$NAR_WORKDIR" e8
python nar/activation_report.py --workdir "$NAR_WORKDIR"
```

For a fresh full extension run on Slurm:

```bash
mkdir -p runs
NAR_WORKDIR=/path/to/project-storage sbatch slurm_extensions.sh
```

Stages have `DONE.json` markers and resumable capture checkpoints. Exact
commands are appended to `runs/commands.jsonl`. Raw bf16 dumps and randomized
eigenspace checkpoints remain under `activations/` in project storage and are
excluded from version control; preserve them for E3/E4.
