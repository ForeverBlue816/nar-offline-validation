#!/bin/bash
#SBATCH --job-name=qwen-full-resolution
#SBATCH --qos=override-limits-but-killable
#SBATCH --constraint=cpu_ok
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runs/qwen-full-%j.out
#SBATCH --error=runs/qwen-full-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="/home/yanlongc/.cache/qwen-full-render-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /projects/nar/nar-validation/venv/bin/python -m nar.activation_viz.full_batch \
  /projects/nar/nar-validation/outputs/qwen_activation_viz/qwen3_8b_seed42 \
  /projects/nar/nar-validation/outputs/qwen_activation_viz/qwen3_8b_seed42_full --workers 8
