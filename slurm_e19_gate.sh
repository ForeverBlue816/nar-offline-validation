#!/bin/bash
#SBATCH --job-name=nar-e19-gate
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --qos=override-limits-but-killable
#SBATCH --time=04:00:00
#SBATCH --output=runs/e19-gate-%j.out
#SBATCH --error=runs/e19-gate-%j.err
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
run() { "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
        --seed "${NAR_E19_SEED:-0}" "$@"; }
run audit
run calibrate
run control
