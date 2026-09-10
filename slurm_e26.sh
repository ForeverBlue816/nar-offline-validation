#!/bin/bash
#SBATCH --job-name=nar-e26
#SBATCH --gpus=4
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e26-%j.out
#SBATCH --error=runs/e26-%j.err
#SBATCH --requeue
# E26 Qwen3-30B-A3B-Base. Stages: audit calibrate control gptq eval finalize.
# NAR_E26_ROTATIONS (space-separated) for gptq; NAR_E26_VARIANT (shrinkage|per_expert|pooled);
# NAR_E26_ROWS / NAR_E26_BENCHMARKS for eval.
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONPATH="${E13_SITEPACKAGES:-$HOME/.e13_packages}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_HUB_CACHE="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
variant="${NAR_E26_VARIANT:-shrinkage}"
run() { "$python_bin" "$code_dir/nar/e26_moe.py" --workdir "$NAR_WORKDIR" --variant "$variant" "$@"; }
for stage in ${NAR_E26_STAGES:-audit calibrate control gptq eval finalize}; do
    echo "########## E26 stage: $stage (variant $variant) ##########"
    case "$stage" in
        audit)     run audit ;;
        calibrate) run calibrate ;;
        control)   run control ${NAR_E26_ROTATIONS:+--rotations ${NAR_E26_ROTATIONS}} ;;
        gptq)      for rotation in ${NAR_E26_ROTATIONS:-hadamard nar_k8 nar_kmax}; do
                       echo "===== gptq $rotation ====="; run gptq --rotation "$rotation"; done ;;
        eval)      for benchmark in ${NAR_E26_BENCHMARKS:-wikitext c4 eight_task}; do
                       for row in ${NAR_E26_ROWS:-bf16 hadamard_asym_g128 nar_k8_asym_g128 nar_kmax_asym_g128}; do
                           echo "===== evaluate $row $benchmark ====="; run evaluate --row "$row" --benchmark "$benchmark"; done; done ;;
        finalize)  run finalize ;;
        *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
done
echo "===== done ====="
