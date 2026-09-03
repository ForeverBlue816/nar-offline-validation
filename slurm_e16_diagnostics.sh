#!/bin/bash
#SBATCH --job-name=nar-e16-diag
#SBATCH --array=0-1
#SBATCH --gpus=pro6000:1
#SBATCH --time=04:00:00
#SBATCH --output=runs/e16-diag-%A_%a.out
#SBATCH --error=runs/e16-diag-%A_%a.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
models=(llama32_3b llama31_8b)
exec "$python_bin" "$code_dir/nar/e16_diagnostics.py" --workdir "$NAR_WORKDIR" \
    --model "${models[$SLURM_ARRAY_TASK_ID]}"
