#!/bin/bash
#SBATCH --job-name=nar-e18-smoke
#SBATCH --gpus=rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=runs/e18-smoke-%j.out
#SBATCH --error=runs/e18-smoke-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$python_bin" "$code_dir/nar/e18_70b.py" --workdir "$NAR_WORKDIR" \
    --model-id unsloth/Llama-3.2-1B --model-key e18_smoke \
    --calibration-sequences 2 --eval-sequences 2 --seq-len 256 \
    --max-layers 2 --oversample 2 --permutation-stride 32 --weight-row-batch 128
