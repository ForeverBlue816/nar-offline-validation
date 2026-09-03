#!/bin/bash
#SBATCH --job-name=nar-e19-qwen3
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e19-qwen3-%A_%a-%j.out
#SBATCH --error=runs/e19-qwen3-%A_%a-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007

code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
sitepackages="${E13_SITEPACKAGES:-$HOME/.e13_packages}"
export PYTHONPATH="$sitepackages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
artifact_root="${NAR_E19_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e19}"
seed="${NAR_E19_SEED:-0}"

# task -> row. Rows 2-5 share one GPTQ configuration; each task owns one row so
# a preempted task only loses its own row, and every artifact write is atomic.
rows=(bf16 hadamard_asym_g128 nar_k8_asym_g128 nar_k32_asym_g128 nar_kmax_asym_g128)
rotations=(none hadamard nar_k8 nar_k32 nar_kmax)
metrics=(ppl both both ppl both)
# Runs as an array (one task per row) when submitted with --array, and as a
# single sequential job otherwise, because the cluster caps submitted jobs per
# user at 5. Either way each row is resumable: GPTQ checkpoints per layer and
# every artifact write is atomic, so a preempted job re-runs only what is
# missing.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    tasks=("$SLURM_ARRAY_TASK_ID")
else
    read -r -a tasks <<< "${NAR_E19_TASKS:-0 1 2 3 4}"
fi

run() { "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
        --artifact-root "$artifact_root" --seed "$seed" "$@"; }

for task in "${tasks[@]}"; do
    row="${rows[$task]}"
    rotation="${rotations[$task]}"
    metric="${metrics[$task]}"
    echo "E19 task=$task row=$row rotation=$rotation metrics=$metric seed=$seed"
    if [[ "$rotation" != none ]]; then
        run gptq --rotation "$rotation"
    fi
    run evaluate --row "$row" --metrics "$metric"
done
"$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
    --artifact-root "$artifact_root" --seed "$seed" finalize
