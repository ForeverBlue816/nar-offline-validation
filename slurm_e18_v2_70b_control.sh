#!/bin/bash
#SBATCH --job-name=nar-70b-ctl
#SBATCH --gpus=pro6000:4
#SBATCH --time=06:00:00
#SBATCH --output=runs/e18v2-70b-ctl-%j.out
#SBATCH --error=runs/e18v2-70b-ctl-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
out="$NAR_WORKDIR/results/llama31_70b"
run() { "$python_bin" "$code_dir/nar/e18_v2.py" --workdir "$NAR_WORKDIR" \
        --model-key llama31_70b --eval-sequences 64 --check-rows 32 --report-only "$@"; }

# The E18 70B row set was produced with the v1 weight fold. Run the control
# both ways: the exact transpose isolates the quantizer, the weight fold
# reproduces the path that produced PPL 15025.
# Each arm is skipped if its artifact already exists, so a rerun only fills in
# what is missing rather than repeating a 70B pass that already succeeded.
if [[ ! -f "$out/e18v2_rotation_only_control_exact.csv" ]]; then
    echo "===== exact-transpose fold ====="
    run --rotation-only-control
    cp -f "$out/e18v2_rotation_only_control.csv" "$out/e18v2_rotation_only_control_exact.csv"
    cp -f "$out/e18v2_fold_audit.json" "$out/e18v2_fold_audit_exact.json"
else
    echo "===== exact-transpose fold already recorded, skipping ====="
fi

if [[ ! -f "$out/e18v2_rotation_only_control_weight_fold.csv" ]]; then
    echo "===== v1 weight fold ====="
    run --rotation-only-control --weight-fold
    cp -f "$out/e18v2_rotation_only_control.csv" "$out/e18v2_rotation_only_control_weight_fold.csv"
    cp -f "$out/e18v2_fold_audit.json" "$out/e18v2_fold_audit_weight_fold.json"
fi
