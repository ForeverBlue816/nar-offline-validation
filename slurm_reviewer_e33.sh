#!/bin/bash
#SBATCH --job-name=nar-e33-rangelaw
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --qos=override-limits-but-killable
#SBATCH --output=runs/reviewer-e33-%j.out
#SBATCH --open-mode=append
#SBATCH --error=runs/reviewer-e33-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME=/projects/nar/nar-validation/cache/huggingface HF_DATASETS_CACHE=/projects/nar/nar-validation/cache/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin=/projects/nar/nar-validation/venv/bin/python
"$python_bin" nar/reviewer_layerwise.py --model "${1:?model}" --smoke
"$python_bin" nar/e33_rangelaw.py --model "$1"
