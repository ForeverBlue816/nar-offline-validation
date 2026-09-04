#!/bin/bash
#SBATCH --job-name=nar-e19-step3
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=12:00:00
#SBATCH --output=runs/e19-step3-%j.out
#SBATCH --error=runs/e19-step3-%j.err
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
main_root="${NAR_E19_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e19}"
# The default-protocol checkpoints on disk predate the audit instrumentation, so
# the diagnostics are regenerated into a separate root. The weights are
# deterministic and identical; only gptq_audit.csv is wanted from this pass, and
# the committed checkpoints are left untouched.
audit_root="$NAR_WORKDIR/artifacts/e19_audit"
seed="${NAR_E19_SEED:-0}"
rotations="hadamard nar_k8 nar_kmax"

run() { "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
        --seed "$seed" "$@"; }

echo "== 3a: default-protocol Hessian and clipping diagnostics =="
for rot in $rotations; do
    run --artifact-root "$audit_root" gptq --rotation "$rot" --protocol default
done

echo "== 3b: alternative protocols, matched across rotations =="
for protocol in act_order g128; do
    for rot in $rotations; do
        run --artifact-root "$main_root" gptq --rotation "$rot" --protocol "$protocol"
    done
done

echo "== 3c: protocol selection on held-out calibration windows =="
for rot in $rotations; do
    row="${rot}_asym_g128"
    [ "$rot" = hadamard ] && row="hadamard_asym_g128"
    run --artifact-root "$main_root" evaluate --row "$row" --metrics ppl --split calibration
done
for protocol in act_order g128; do
    for rot in $rotations; do
        run --artifact-root "$main_root" evaluate --row "${rot}_${protocol}" \
            --metrics ppl --split calibration
    done
done
echo "== step 3 runs complete =="
