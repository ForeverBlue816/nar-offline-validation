#!/bin/bash
#SBATCH --job-name=nar-e14-bf16
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=00:40:00
#SBATCH --output=runs/e14-bf16-%j.out
#SBATCH --error=runs/e14-bf16-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007

code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
sitepackages="${E13_SITEPACKAGES:-$HOME/.e13_packages}"
export PYTHONPATH="$sitepackages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"

# 141 windows is what the E14 rows scored; a different count means the
# reference is measuring a different token stream and must not be reported.
"$python_bin" "$code_dir/nar/e14_bf16_reference.py" \
    --workdir "$NAR_WORKDIR" --model "${NAR_E14_BF16_MODEL:-llama31_8b}" \
    --seq-len 2048 --seed "${NAR_E14_BF16_SEED:-0}" --expect-chunks ${NAR_E14_BF16_CHUNKS:-141}
