# NAR offline validation

`experiment.py` reproduces the frozen E0, E1, and E2 tables. The extension
`extended_experiment.py` reads those outputs without rerunning them, performs
E1b, captures the wide activations once, and evaluates E1c/E1d.
`activation_experiments.py` adds activation-only perplexity and factorized
online-cost checks (E5/E6). `activation_diagnostics.py` adds the per-token V
cache and one-shot range-direct diagnostics (E7/E8), while
`activation_report.py` appends their tables and figures to the corrected
report. `e11_fair_baselines.py` adds the frozen SmoothQuant, DuQuant-style,
bit-fair, group-size, and rank-knee comparisons. `e12_wy.py` benchmarks compact
WY R4, and `e13_zero_shot.py` runs the pinned zero-shot transfer suite.
`e14_w4a4kv4.py` adds the requested QuaRot-GPTQ end-to-end rows, while
`e15_fp4.py` tests the no-zero-point FP4 boundary on the retained E1c tensors.

E1--E13 are forward-hook/offline/fake-quant experiments. E14 is the explicitly
requested expansion to GPTQ W4A4KV4; its large layer checkpoints stay outside
Git. All comparisons are paired and no result-driven tuning is used.

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
- E11 reuses E5 bf16/Hadamard/NAR rows verbatim and adds both-site activation
  baselines on 3B/8B. SmoothQuant uses alpha=0.5. The DuQuant-style row uses the
  pinned upstream construction audit and one requested outlier direction per
  block. Group/rank choices are fixed at g=64/128/256 and k=8/16/32/kmax.
- E12 uses compact WY `G=I-WY^T` at k=16/32/64 and times tokens=1/32/2048
  against the same unfused Hadamard and bf16 down_proj matmul as E6.
- E13 uses lm-evaluation-harness commit
  `b954108c9baaaa934b4ad842033b31a97ee30816`, zero-shot, seed 20260902, on
  PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande, and LAMBADA.
- E14 ports GPTQ and its MSE clipping search from spcl/QuaRot commit
  `5008669b08c1f11f9b64d52d16fddd47ca754c5a`: symmetric W4 per output
  channel (`groupsize=-1`), block 128, damp 0.01, no act order, 128 fixed
  WikiText-2 sequences. Embeddings/lm_head remain bf16. All rows replace R3 K
  with asymmetric KIVI-style per-channel K4 over 32-token groups and the
  standard 128-token residual policy; V is asymmetric per-token V4 with a
  128-token bf16 residual. NAR R4 uses the E12 compact-WY representation.
- E15 uses nearest E2M1 with one E4M3FN max/6 scale per block of 16 and no
  zero-point. Its Hadamard baseline is a fixed random-sign H16 aligned to each
  scale block. It samples every 128th retained E1c token and includes one fixed
  condition-16 log-diagonal non-Gaussian invertible control.

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

Fair-baseline, deployment, and transfer continuation:

```bash
python nar/e11_fair_baselines.py --workdir "$NAR_WORKDIR" calibrate --model llama32_3b
python nar/e11_fair_baselines.py --workdir "$NAR_WORKDIR" evaluate --model llama32_3b
python nar/e11_fair_baselines.py --workdir "$NAR_WORKDIR" calibrate --model llama31_8b
python nar/e11_fair_baselines.py --workdir "$NAR_WORKDIR" evaluate --model llama31_8b
python nar/e11_fair_baselines.py --workdir "$NAR_WORKDIR" decision
python nar/e11_report.py --workdir "$NAR_WORKDIR"
python nar/e12_wy.py --workdir "$NAR_WORKDIR"
python nar/e12_report.py --workdir "$NAR_WORKDIR"
pip install -r requirements-e13.txt
python nar/e13_zero_shot.py --workdir "$NAR_WORKDIR" run --model llama32_3b --method bf16
python nar/e13_zero_shot.py --workdir "$NAR_WORKDIR" run --model llama32_3b --method hadamard
python nar/e13_zero_shot.py --workdir "$NAR_WORKDIR" run --model llama32_3b --method nar
python nar/e13_zero_shot.py --workdir "$NAR_WORKDIR" finalize --model llama32_3b
```

End-to-end and FP4 boundary stages (run E14 only because the E11 stop condition did not fire):

```bash
python nar/e14_w4a4kv4.py --workdir "$NAR_WORKDIR" calibrate --model llama32_3b
python nar/e14_w4a4kv4.py --workdir "$NAR_WORKDIR" calibrate --model llama31_8b
python nar/e14_w4a4kv4.py --workdir "$NAR_WORKDIR" gptq --model llama32_3b --rotation hadamard
python nar/e14_w4a4kv4.py --workdir "$NAR_WORKDIR" gptq --model llama32_3b --rotation nar
python nar/e14_w4a4kv4.py --workdir "$NAR_WORKDIR" evaluate --model llama32_3b --row quarot
python nar/e14_w4a4kv4.py --workdir "$NAR_WORKDIR" evaluate --model llama32_3b --row hadamard_asym_g128
python nar/e14_w4a4kv4.py --workdir "$NAR_WORKDIR" evaluate --model llama32_3b --row nar_asym_g128
python nar/e14_w4a4kv4.py --workdir "$NAR_WORKDIR" finalize
python nar/e14_report.py --workdir "$NAR_WORKDIR"
python nar/e15_fp4.py --workdir "$NAR_WORKDIR"
python nar/e15_report.py --workdir "$NAR_WORKDIR"
```

For a fresh full extension run on Slurm:

```bash
mkdir -p runs
NAR_WORKDIR=/path/to/project-storage sbatch slurm_extensions.sh
```

Stages have `DONE.json` markers and resumable capture checkpoints. Exact
commands are appended to `runs/commands.jsonl`. Raw bf16 dumps and randomized
eigenspace checkpoints remain under `activations/` in project storage and are
excluded from version control; preserve them for E3/E4. E14 GPTQ checkpoints
default to `$NAR_WORKDIR/artifacts/e14` and are likewise not committed.
