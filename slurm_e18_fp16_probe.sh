#!/bin/bash
#SBATCH --job-name=nar-fp16-probe
#SBATCH --gpus=pro6000:4
#SBATCH --time=02:00:00
#SBATCH --output=runs/fp16-probe-%j.out
#SBATCH --error=runs/fp16-probe-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
"$NAR_WORKDIR/venv/bin/python" "$code_dir/nar/e18_fp16_range_probe.py" \
    --workdir "$NAR_WORKDIR" --model-key "${NAR_PROBE_MODEL:-llama31_70b}" --sequences 4
