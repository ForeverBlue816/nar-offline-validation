#!/bin/bash
#SBATCH --job-name=nar-e14-fold
#SBATCH --gpus=rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=runs/e14-fold-%j.out
#SBATCH --error=runs/e14-fold-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$python_bin" "$code_dir/nar/e14_fold_diagnostic.py" --workdir "$NAR_WORKDIR" \
    --model "${NAR_MODEL:-llama32_3b}" --rotation "${NAR_ROTATION:-hadamard}" \
    --model-dtype "${NAR_MODEL_DTYPE:-bf16}" --tokens "${NAR_VERIFY_TOKENS:-128}"
