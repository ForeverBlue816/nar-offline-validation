#!/bin/bash
#SBATCH --job-name=nar-e17-v2-fold
#SBATCH --gpus=pro6000:1
#SBATCH --qos=override-limits-but-killable
#SBATCH --time=02:00:00
#SBATCH --output=runs/e17-v2-fold-%j.out
#SBATCH --error=runs/e17-v2-fold-%j.err
set -euo pipefail
umask 0007

code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
sitepackages="${E13_SITEPACKAGES:-$HOME/.e13_packages}"
export PYTHONPATH="$sitepackages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
exec "$python_bin" "$code_dir/nar/e17_v2_fold_verify.py" --workdir "$NAR_WORKDIR" "$@"
