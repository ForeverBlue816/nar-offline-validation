# NAR offline validation

`experiment.py` is the single entry point that reproduces E0, all E1 tables
and figures, both E2 model tables, confidence intervals, and `report.md`.

The experiment is intentionally limited to forward hooks, offline tensors, and
KV-only fake quantization. It never modifies weights, runs GPTQ, or builds a
W4A4 pipeline.

## One-command reproduction

From the project root on a Slurm GPU allocation:

```bash
./run_all.sh
```

Or submit the frozen batch script from a login node:

```bash
sbatch slurm.sh
```

Stages are checkpointed with `DONE.json`/`E1_DONE.json`/`E2_DONE.json`. Running
the command again resumes at the first incomplete stage. Exact commands are
appended to `runs/commands.jsonl`, while the Slurm transcript and per-stage
logs live in `runs/`.

## Fixed choices

- Llama weights: public exact-weight mirrors `unsloth/Llama-3.2-3B` and
  `unsloth/Llama-3.2-1B`; no fine-tuned model is used.
- E1 calibration A: first 128 BOS-prefixed, length-2048 chunks of WikiText-2
  train. Calibration B: the next disjoint 128 chunks.
- Every token and KV head contributes to the uncentered second moment.
  Stored offline tensors deterministically subsample positions
  `0,32,...,2016` from all 128 sequences, all attention heads for Q, and all KV heads for K.
- E2: first 64 BOS-prefixed, length-2048 WikiText-2 test chunks, three rotation
  seeds, paired methods. The 3B model uses b=64 and b=128; the 1B model uses
  b=32 because its head dimension is 64.
- NAR-RoPE requires an even number of at least two groups. It is undefined at
  3B b=128 and 1B b=64; the code reports N/A rather than substituting identity.
- “Substantial” top-k range attribution is fixed at 10%, since the hypothesis
  prompt did not provide a numeric threshold. Held-out stability is fixed as a
  positive reduction retaining at least 80% of the calibration-A reduction.

## Commands

```bash
python nar/experiment.py --workdir . e0
python nar/experiment.py --workdir . collect --model llama32_3b --tag cal_a
python nar/experiment.py --workdir . analyze --model llama32_3b
python nar/experiment.py --workdir . e2 --model llama32_3b
python nar/experiment.py --workdir . report
```

The cache, model shards, activation dumps, results, and environment are written
under the selected `--workdir` and are excluded from version control.
