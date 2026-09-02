#!/bin/bash
#SBATCH --job-name=nar-e11-8b
#SBATCH --gpus=rtx5090:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e11-8b-%j.out
#SBATCH --error=runs/e11-8b-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to project storage}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" MPLCONFIGDIR="$NAR_WORKDIR/cache/matplotlib" TMPDIR="$NAR_WORKDIR/tmp"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$NAR_WORKDIR/runs" "$TMPDIR" "$MPLCONFIGDIR"
"$python_bin" "$code_dir/nar/e11_fair_baselines.py" --workdir "$NAR_WORKDIR" calibrate --model llama31_8b --batch-size 2
exec "$python_bin" "$code_dir/nar/e11_fair_baselines.py" --workdir "$NAR_WORKDIR" evaluate --model llama31_8b
