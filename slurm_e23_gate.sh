#!/bin/bash
#SBATCH --job-name=nar-e23-gate
#SBATCH --gpus=2
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/e23-%j.out
#SBATCH --error=runs/e23-%j.err
#SBATCH --requeue
# E23 step 2: reproduce the published 16-bit rows. One job, every candidate
# checkpoint in turn; each gate skips its own finished artifact on requeue.
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
: "${NAR_E23_MODELS:?Set NAR_E23_MODELS (+-separated Hugging Face ids)}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONPATH="${E13_SITEPACKAGES:-$HOME/.e13_packages}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_HUB_CACHE="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
for model in ${NAR_E23_MODELS//+/ }; do
    echo "########## E23 gate: $model ##########"
    "$python_bin" "$code_dir/nar/e23_bridge.py" --workdir "$NAR_WORKDIR" gate --model-id "$model" ${NAR_E23_GATE_ARGS:-}
done
echo "===== done ====="
