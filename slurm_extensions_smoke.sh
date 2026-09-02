#!/bin/bash
#SBATCH --job-name=nar-hook-smoke
#SBATCH --gpus=6000ada:1
#SBATCH --time=00:20:00
#SBATCH --output=runs/extensions-smoke-%j.out
#SBATCH --error=runs/extensions-smoke-%j.err

set -euo pipefail
umask 0007

code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to a project-storage path}"
work_dir="$NAR_WORKDIR"
export HF_HOME="$work_dir/cache/huggingface"
export HF_DATASETS_CACHE="$work_dir/cache/datasets"
export XDG_CACHE_HOME="$work_dir/cache/xdg"
export MPLCONFIGDIR="$work_dir/cache/matplotlib"
export TMPDIR="$work_dir/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$work_dir/runs" "$TMPDIR" "$MPLCONFIGDIR"
exec "$work_dir/venv/bin/python" "$code_dir/nar/extended_experiment.py" --workdir "$work_dir" \
  collect-wide --tag wide_smoke --sequences 2 --seq-len 64 --batch-size 1 --checkpoint-sequences 1
