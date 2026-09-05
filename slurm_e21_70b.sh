#!/bin/bash
#SBATCH --job-name=nar-e21-70b
#SBATCH --gpus=pro6000:8
#SBATCH --constraint=highmem
#SBATCH --time=12:00:00
#SBATCH --output=runs/e21-70b-%j.out
#SBATCH --error=runs/e21-70b-%j.err
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
run() { "$python_bin" "$code_dir/nar/e21_llama70b_e2e.py" --workdir "$NAR_WORKDIR" \
        --seed "${NAR_E21_SEED:-0}" "$@"; }

for stage in ${NAR_E21_STAGES:-audit calibrate}; do
    echo "===== E21 stage: $stage ====="
    case "$stage" in
        audit)     run audit ;;
        calibrate) run calibrate ;;
        bf16)      run evaluate --row bf16 ;;
        eval_k8)   run evaluate --row nar_k8_asym_g128 ;;
        eval_kmax) run evaluate --row nar_kmax_asym_g128 ;;
        finalize)  run finalize ;;
        *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
done
