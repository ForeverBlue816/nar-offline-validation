#!/bin/bash
#SBATCH --job-name=qwen-local-typography
#SBATCH --qos=override-limits-but-killable
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/qwen-local-typography-%j.out
#SBATCH --error=runs/qwen-local-typography-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${QWEN_FIGURE_DEPS:+$QWEN_FIGURE_DEPS:}$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /projects/nar/nar-validation/venv/bin/python -u -m nar.activation_viz.local_typography_revision \
  /projects/nar/nar-validation/outputs/qwen_activation_viz/qwen3_8b_seed42 \
  outputs/qwen_activation_viz/qwen3_8b_seed42 "$@"
