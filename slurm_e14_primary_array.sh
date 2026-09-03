#!/bin/bash
#SBATCH --job-name=nar-e14-primary
#SBATCH --gpus=rtx5090:1
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/e14-primary-%A_%a.out
#SBATCH --error=runs/e14-primary-%A_%a.err
#SBATCH --requeue
set -euo pipefail
umask 0007

code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"

models=(llama32_3b llama31_8b)
rotations=(hadamard nar_k8 nar_kmax)
rows=(hadamard_asym_g128 nar_k8_asym_g128 nar_kmax_asym_g128)
task="$SLURM_ARRAY_TASK_ID"
model_index=$((task / 9))
remainder=$((task % 9))
seed=$((remainder / 3))
rotation_index=$((remainder % 3))
model="${models[$model_index]}"
rotation="${rotations[$rotation_index]}"
row="${rows[$rotation_index]}"

artifact_root="${NAR_E14_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e14}"
python_bin="$NAR_WORKDIR/venv/bin/python"
sitepackages="${E13_SITEPACKAGES:-$HOME/.e13_packages}"
export PYTHONPATH="$sitepackages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

echo "E14 task=$task model=$model rotation=$rotation row=$row seed=$seed"
"$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
    --artifact-root "$artifact_root" --seed "$seed" gptq \
    --model "$model" --rotation "$rotation"
"$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
    --artifact-root "$artifact_root" --seed "$seed" evaluate \
    --model "$model" --row "$row" --metrics both

# The released symmetric-A row reuses the seed-0 Hadamard GPTQ checkpoint.
if [[ "$rotation" == hadamard && "$seed" == 0 ]]; then
    "$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
        --artifact-root "$artifact_root" --seed 0 evaluate \
        --model "$model" --row quarot_released --metrics both
fi
