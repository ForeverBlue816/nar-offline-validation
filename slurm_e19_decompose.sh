#!/bin/bash
#SBATCH --job-name=nar-e19-decomp
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=08:00:00
#SBATCH --output=runs/e19-decomp-%j.out
#SBATCH --error=runs/e19-decomp-%j.err
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

run() { "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
        --artifact-root "$artifact_root" --seed "$seed" "$@"; }

# KV-only needs no GPTQ; W-only reuses the layer states already on disk. Each
# row is skipped if its artifact exists, so a preempted job resumes for free.
rows="${NAR_E19_DECOMP_ROWS:-kv_only_hadamard kv_only_nar_k8 kv_only_nar_kmax w_only_hadamard w_only_nar_k8 w_only_nar_kmax}"
for row in $rows; do
    echo "E19 decomposition row=$row seed=$seed"
    run evaluate --row "$row" --metrics ppl
done
run decompose
