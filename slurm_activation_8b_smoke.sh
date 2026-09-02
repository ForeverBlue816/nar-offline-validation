#!/bin/bash
#SBATCH --job-name=nar-8b-smoke
#SBATCH --gpus=rtx5090:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/activation-8b-smoke-%j.out
#SBATCH --error=runs/activation-8b-smoke-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to project storage}"
work_dir="$NAR_WORKDIR"
smoke="$work_dir/smoke_activation"
python_bin="$work_dir/venv/bin/python"
export HF_HOME="$work_dir/cache/huggingface" HF_DATASETS_CACHE="$work_dir/cache/datasets"
export XDG_CACHE_HOME="$work_dir/cache/xdg" MPLCONFIGDIR="$work_dir/cache/matplotlib" TMPDIR="$work_dir/tmp"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$work_dir/runs" "$smoke/runs" "$smoke/results" "$TMPDIR" "$MPLCONFIGDIR"
"$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$smoke" calibrate --model llama31_8b --calibration-sequences 2 --seq-len 128 --batch-size 1
exec "$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$smoke" e5 --model llama31_8b --eval-sequences 1 --seq-len 128 --seeds 1 --weight-row-batch 256
