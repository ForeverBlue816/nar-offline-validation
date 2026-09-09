#!/bin/bash
#SBATCH --job-name=nar-e22
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e22-%j.out
#SBATCH --error=runs/e22-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
: "${NAR_E22_MODEL:?Set NAR_E22_MODEL}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONPATH="${E13_SITEPACKAGES:-$HOME/.e13_packages}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${NAR_HF_OFFLINE:-1}" HF_DATASETS_OFFLINE="${NAR_HF_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
model="$NAR_E22_MODEL"
run() { "$python_bin" "$code_dir/${NAR_E22_SCRIPT:-nar/e22_qwen3_family.py}" --workdir "$NAR_WORKDIR" --model "$model" \
        ${NAR_E22_BATCH:+--batch-size "$NAR_E22_BATCH"} ${NAR_E22_EXTRA:-} "$@"; }

# Every step skips its own completed artifact, so a requeue resumes.
for stage in ${NAR_E22_STAGES:-audit calibrate control gptq eval finalize}; do
    echo "########## E22 $model stage: $stage ##########"
    case "$stage" in
        audit)     run audit ;;
        calibrate) run calibrate ;;
        control)   run control ;;
        gptq)      for rotation in ${NAR_E22_ROTATIONS:-hadamard nar_k8 nar_kmax}; do
                       echo "===== gptq $rotation ====="; run gptq --rotation "$rotation"; done ;;
        gate)      run gate --items "${NAR_E22_GATE_ITEMS:-100}" --gate-batch "${NAR_E22_GATE_BATCH:-8}" ;;
        eval)      for benchmark in ${NAR_E22_BENCHMARKS:-wikitext c4 mmlu gsm8k mmlu_redux arc_easy}; do
                       for row in ${NAR_E22_ROWS:-bf16 hadamard_asym_g128 nar_k8_asym_g128 nar_kmax_asym_g128}; do
                           echo "===== evaluate $row $benchmark ====="
                           run evaluate --row "$row" --benchmark "$benchmark"
                       done
                   done ;;
        finalize)  run finalize ;;
        *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
done
echo "===== done ====="
