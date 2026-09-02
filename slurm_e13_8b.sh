#!/bin/bash
#SBATCH --job-name=nar-e13-8b
#SBATCH --gpus=rtx5090:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e13-8b-%j.out
#SBATCH --error=runs/e13-8b-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to project storage}"
python_bin="$NAR_WORKDIR/venv/bin/python"
sitepackages="${E13_SITEPACKAGES:-$HOME/.e13_packages}"
export PYTHONPATH="$sitepackages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for method in bf16 hadamard nar; do
    "$python_bin" "$code_dir/nar/e13_zero_shot.py" --workdir "$NAR_WORKDIR" run --model llama31_8b --method "$method"
done
exec "$python_bin" "$code_dir/nar/e13_zero_shot.py" --workdir "$NAR_WORKDIR" finalize --model llama31_8b
