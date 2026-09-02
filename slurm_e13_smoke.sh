#!/bin/bash
#SBATCH --job-name=nar-e13-smoke
#SBATCH --gpus=rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=runs/e13-smoke-%j.out
#SBATCH --error=runs/e13-smoke-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to project storage}"
python_bin="$NAR_WORKDIR/venv/bin/python"
smoke_dir="$NAR_WORKDIR/smoke_e13"
sitepackages="${E13_SITEPACKAGES:-$HOME/.e13_packages}"
export PYTHONPATH="$sitepackages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$smoke_dir"
exec "$python_bin" "$code_dir/nar/e13_zero_shot.py" --workdir "$smoke_dir" --asset-workdir "$NAR_WORKDIR" run --model llama32_3b --method bf16 --batch-size 1 --limit 1
