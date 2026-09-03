#!/bin/bash
#SBATCH --job-name=nar-e18-v2
#SBATCH --gpus=pro6000:1
#SBATCH --qos=override-limits-but-killable
#SBATCH --time=12:00:00
#SBATCH --output=runs/e18-v2-%j.out
#SBATCH --error=runs/e18-v2-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
exec "$python_bin" "$code_dir/nar/e18_v2.py" --workdir "$NAR_WORKDIR" "$@"
