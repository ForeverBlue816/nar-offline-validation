#!/bin/bash
#SBATCH --job-name=nar-kv425
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/kv425-%j.out
#SBATCH --error=runs/kv425-%j.err
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

# K_TOKEN_GROUP 128 makes each quantized K value 4.25 bits, matching V. This is
# a property of the cache at runtime: no weight changes and no GPTQ output is
# re-derived, so only the evaluations re-run, and they write to _kg128 artifacts
# so the frozen K=32 rows are untouched.
for model in ${NAR_KV_MODELS:-llama32_3b llama31_8b}; do
    for row in hadamard_asym_g128 nar_k8_asym_g128 nar_kmax_asym_g128; do
        echo "===== E14 $model $row, K token group 128 ====="
        "$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
            --artifact-root "${NAR_E14_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e14}" --seed 0 \
            evaluate --model "$model" --row "$row" --metrics both --k-token-group 128
    done
done

if [ "${NAR_KV_QWEN:-1}" = 1 ]; then
    for row in hadamard_asym_g128 nar_k8_asym_g128 nar_kmax_asym_g128; do
        echo "===== E19 $row, K token group 128 ====="
        "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
            --artifact-root "${NAR_E19_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e19}" --seed 0 \
            evaluate --row "$row" --metrics both --k-token-group 128
    done
fi
echo "===== KV 4.25-bit rows complete ====="
