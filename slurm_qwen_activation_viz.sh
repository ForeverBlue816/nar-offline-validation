#!/bin/bash
#SBATCH --job-name=qwen8b-activation-viz
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --qos=override-limits-but-killable
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --output=runs/qwen-activation-viz-%j.out
#SBATCH --error=runs/qwen-activation-viz-%j.err
set -euo pipefail
umask 0007
cd "${SLURM_SUBMIT_DIR}"
export NAR_WORKDIR=/projects/nar/nar-validation
export HF_HOME="$NAR_WORKDIR/cache/huggingface"
export HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HOME/.e13_packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$NAR_WORKDIR/venv/bin/python" -m nar.activation_viz.capture --workdir "$NAR_WORKDIR" --output "$NAR_WORKDIR/outputs/qwen_activation_viz/qwen3_8b_seed42"
