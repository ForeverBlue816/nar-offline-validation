#!/bin/bash
#SBATCH --job-name=nar-e14-eval
#SBATCH --gpus=rtx5090:1
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/e14-eval-%j.out
#SBATCH --error=runs/e14-eval-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
: "${NAR_MODEL:?Set NAR_MODEL}"
: "${NAR_ROW:?Set NAR_ROW}"
artifact_root="${NAR_E14_ARTIFACT_ROOT:-$HOME/nar-e14-artifacts}"
python_bin="$NAR_WORKDIR/venv/bin/python"
sitepackages="${E13_SITEPACKAGES:-$HOME/.e13_packages}"
export PYTHONPATH="$sitepackages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
    --artifact-root "$artifact_root" evaluate --model "$NAR_MODEL" --row "$NAR_ROW"
