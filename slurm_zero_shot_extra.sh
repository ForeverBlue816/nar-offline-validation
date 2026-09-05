#!/bin/bash
#SBATCH --job-name=nar-zs-extra
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/zs-extra-%j.out
#SBATCH --error=runs/zs-extra-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONPATH="${E13_SITEPACKAGES:-$HOME/.e13_packages}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"

# boolq, openbookqa and social_iqa only, into their own artifact. The frozen six
# are never re-run: --task-set extra selects a disjoint task list and a distinct
# filename, and --metrics zero_shot means no perplexity is recomputed either.
for model in ${NAR_ZS_MODELS:-llama32_3b llama31_8b}; do
    for row in quarot_released hadamard_asym_g128 nar_k8_asym_g128 nar_kmax_asym_g128; do
        echo "===== E14 $model $row (extra tasks) ====="
        "$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
            --artifact-root "${NAR_E14_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e14}" --seed 0 \
            evaluate --model "$model" --row "$row" --metrics zero_shot --task-set extra
    done
done

if [ "${NAR_ZS_QWEN:-1}" = 1 ]; then
    for row in bf16 hadamard_asym_g128 nar_k8_asym_g128 nar_k32_asym_g128 nar_kmax_asym_g128; do
        echo "===== E19 $row (extra tasks) ====="
        "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
            --artifact-root "${NAR_E19_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e19}" --seed 0 \
            evaluate --row "$row" --metrics zero_shot --task-set extra --all-rows-zero-shot
    done
    # bf16 and nar_k32 have no frozen-six artifact, so the eight-task mean needs
    # those too; the six are computed here for the first time, not re-run.
    for row in bf16 nar_k32_asym_g128; do
        echo "===== E19 $row (frozen six, first measurement) ====="
        "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
            --artifact-root "${NAR_E19_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e19}" --seed 0 \
            evaluate --row "$row" --metrics zero_shot --all-rows-zero-shot
    done
fi
echo "===== extra-task zero-shot complete ====="
