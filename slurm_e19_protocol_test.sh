#!/bin/bash
#SBATCH --job-name=nar-e19-ptest
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=03:00:00
#SBATCH --output=runs/e19-ptest-%j.out
#SBATCH --error=runs/e19-ptest-%j.err
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
run() { "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
        --artifact-root "${NAR_E19_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e19}" \
        --seed "${NAR_E19_SEED:-0}" "$@"; }
# Both alternative protocols are reported on test. The protocol choice was made
# on the calibration windows and is not revised by anything below.
for protocol in act_order g128; do
    for rot in hadamard nar_k8 nar_kmax; do
        run evaluate --row "${rot}_${protocol}" --metrics ppl
    done
done
