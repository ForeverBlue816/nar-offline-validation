#!/bin/bash
#SBATCH --job-name=nar-e6-e8
#SBATCH --gpus=rtx5090:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e6-e8-%j.out
#SBATCH --error=runs/e6-e8-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to project storage}"
work_dir="$NAR_WORKDIR"
python_bin="$work_dir/venv/bin/python"
export HF_HOME="$work_dir/cache/huggingface" HF_DATASETS_CACHE="$work_dir/cache/datasets"
export XDG_CACHE_HOME="$work_dir/cache/xdg" MPLCONFIGDIR="$work_dir/cache/matplotlib" TMPDIR="$work_dir/tmp"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$work_dir/runs" "$TMPDIR" "$MPLCONFIGDIR"
"$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$work_dir" calibrate --model llama32_3b
"$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$work_dir" e6
"$python_bin" "$code_dir/nar/activation_diagnostics.py" --workdir "$work_dir" collect-v
"$python_bin" "$code_dir/nar/activation_diagnostics.py" --workdir "$work_dir" e7
"$python_bin" "$code_dir/nar/activation_diagnostics.py" --workdir "$work_dir" collect-down-heldout
exec "$python_bin" "$code_dir/nar/activation_diagnostics.py" --workdir "$work_dir" e8
