# E27 execution

The preregistration is committed before any E27 GPU measurement. `e27_preregistration.json`
contains token-cache and A-factor hashes, the user-confirmed global top-1% rule, fixed
seeds/ranks, and the conditional D trigger. The E27 section of `report.md` records H-A/H-B.

Run both models sequentially on one GPU:

```bash
NAR_WORKDIR=/projects/nar/nar-validation sbatch slurm_e27.sh
```

The job first runs the seven numerical tests, checks frozen inputs, and performs a GPU
smoke test of fp32 NLL, capture, and exact-transpose folding. It then captures required
rows, builds B/C and BOS-excluded factors with A's fixed permutation, measures per-layer
energy/range/NMSE/angles, and evaluates all three activation-site conditions with three
paired seeds. A and Hadamard use the same evaluation hardware. If a B variant has lower
mean PPL in any condition, the preregistered E8-style D runs automatically for that model.

Stages are resumable; checkpoints are per layer/factor and every eight PPL sequences.
A partially completed capture is replayed from its frozen token IDs. Raw activation rows,
selection indices/values, and rotation factors stay in project storage under
`activations/<model>/e27/`, not Git. Original E1c files and A factors are read-only.
Numerical rank is recorded because top-1 selection can consist of repeated BOS values;
angle rows explicitly flag when the eight directions include zero-energy completion.

Completed files are copied into `results/<model>/`: `e27_summary.csv`,
`e27_per_sequence.csv`, `e27_f_and_angles.csv`, `e27_per_layer.csv`, selection/factor
and baseline-replay audits, `e27_law_fit.csv`, and `E27_DONE.json`. A completion marker
requires every expected method/site/layer/seed, including D when triggered. The main
report is updated, with a standalone copy in `results/e27_report.md`.

At initial submission the account's permitted Slurm queues had no remaining submission
slots. `scripts/submit_e27_when_available.py` retries once per minute, under a process
lock, without cancelling or modifying another experiment. `experiments/e27_execution.json`
records its latest state and eventual job ID. `scripts/watch_e27_completion.py` then
tracks that job, verifies completed-file hashes, and pushes E27 outputs to GitHub main
under the user's existing authorization. It never marks a failed or incomplete job done.
Unrelated report edits are excluded from automatic commits; the standalone E27 report
is always included. Logs are `runs/e27-submission-watch.log`,
`runs/e27-completion-watch.log`, and `runs/e27-<jobid>.out/.err`.
