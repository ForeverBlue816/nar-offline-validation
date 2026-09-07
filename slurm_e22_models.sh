#!/bin/bash
#SBATCH --job-name=nar-e22-multi
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e22-%j.out
#SBATCH --error=runs/e22-%j.err
#SBATCH --requeue
# Run slurm_e22_model.sh for several models in one allocation (the QoS caps
# submitted jobs, not work per job). Models are separated by "+" because
# sbatch --export splits values on commas.
set -euo pipefail
: "${NAR_E22_MODELS:?Set NAR_E22_MODELS, e.g. qwen3_0.6b_base+qwen3_1.7b_base}"
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
for model in ${NAR_E22_MODELS//+/ }; do
    echo "########## E22 multi: $model ##########"
    NAR_E22_MODEL="$model" bash "$code_dir/slurm_e22_model.sh"
done
echo "===== multi done ====="
