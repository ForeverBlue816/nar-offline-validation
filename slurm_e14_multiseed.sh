#!/bin/bash
#SBATCH --job-name=nar-e14-seeds
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e14-seeds-%j.out
#SBATCH --error=runs/e14-seeds-%j.err
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
model="${NAR_SEED_MODEL:-llama31_8b}"
artifact_root="${NAR_E14_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e14}"

run() { "$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
        --artifact-root "$artifact_root" "$@"; }

# E14 was amended to a single seed, so every paired delta it reports has no
# confidence interval. These seeds supply one. The seed varies the random sign
# vectors inside each rotation, the randomized sketch that finds the
# eigendirections, and the GPTQ calibration sample, so it exercises every
# stochastic element the pipeline has. quarot_released is excluded: it is a
# released checkpoint, not a re-run of ours, and is seed 0 by definition.
for seed in ${NAR_SEEDS:-1 2}; do
    echo "===== seed $seed: rotation calibration ====="
    run --seed "$seed" calibrate --model "$model"
    for rotation in hadamard nar_k8 nar_kmax; do
        echo "===== seed $seed: GPTQ $rotation ====="
        run --seed "$seed" gptq --model "$model" --rotation "$rotation" --calibration-seed "$seed"
    done
    for row in hadamard_asym_g128 nar_k8_asym_g128 nar_kmax_asym_g128; do
        echo "===== seed $seed: evaluate $row ====="
        run --seed "$seed" evaluate --model "$model" --row "$row" --metrics both
    done
done
echo "===== finalize across seeds ====="
run --seed 0 finalize --seeds $(( 1 + $(echo ${NAR_SEEDS:-1 2} | wc -w) )) || true
