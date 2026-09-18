#!/bin/bash
#SBATCH --job-name=nar-reviewer-eval
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --qos=override-limits-but-killable
#SBATCH --output=runs/reviewer-eval-%j.out
#SBATCH --error=runs/reviewer-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HOME=/projects/nar/nar-validation/cache/huggingface
export HF_DATASETS_CACHE=/projects/nar/nar-validation/cache/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin=/projects/nar/nar-validation/venv/bin/python
model="${1:?model}"
shift
for exp in "$@"; do
  if [[ "$exp" == 30 ]]; then "$python_bin" nar/prepare_e30.py; fi
  "$python_bin" nar/run_reviewer_experiments.py --experiment "$exp" --model "$model"
done
