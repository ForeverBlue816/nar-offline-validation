#!/bin/bash
#SBATCH --job-name=nar-e18-70b-4g
#SBATCH --gpus=pro6000:4
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/e18-70b-4gpu-%j.out
#SBATCH --error=runs/e18-70b-4gpu-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$python_bin" "$code_dir/nar/e18_70b.py" --workdir "$NAR_WORKDIR" \
    --model-id unsloth/Meta-Llama-3.1-70B --model-key llama31_70b \
    --calibration-sequences 128 --eval-sequences 64 --seq-len 2048 \
    --batch-size 1 --oversample 16 --permutation-stride 32 --weight-row-batch 256
