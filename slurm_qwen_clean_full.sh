#!/bin/bash
#SBATCH --job-name=qwen-clean-full
#SBATCH --qos=override-limits-but-killable
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/qwen-clean-full-%j.out
#SBATCH --error=runs/qwen-clean-full-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${QWEN_FIGURE_DEPS:+$QWEN_FIGURE_DEPS:}$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /projects/nar/nar-validation/venv/bin/python -u -m nar.activation_viz.clean_full_revision \
  /projects/nar/nar-validation/outputs/qwen_activation_viz/qwen3_8b_seed42 \
  outputs/qwen_activation_viz/qwen3_8b_seed42 "$@"
