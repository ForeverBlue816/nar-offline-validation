#!/bin/bash
#SBATCH --job-name=qwen8b-activation-render
#SBATCH --qos=override-limits-but-killable
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=runs/qwen-render-%j.out
#SBATCH --error=runs/qwen-render-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /projects/nar/nar-validation/venv/bin/python -m nar.activation_viz.render_batch \
  /projects/nar/nar-validation/outputs/qwen_activation_viz/qwen3_8b_seed42 --workers 8
