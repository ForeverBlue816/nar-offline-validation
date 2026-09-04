#!/bin/bash
#SBATCH --job-name=prism-fig2-f
#SBATCH --array=0-1
#SBATCH --gpus=rtx5090:1
#SBATCH --time=08:00:00
#SBATCH --output=runs/fig2-null-energy-%A_%a.out
#SBATCH --error=runs/fig2-null-energy-%A_%a.err
#SBATCH --requeue
set -euo pipefail
umask 0007

code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
output_dir="${NAR_FIGURE_STATS_DIR:-$NAR_WORKDIR/figure_stats}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface"
export HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
models=(llama32_3b llama31_8b)

exec "$python_bin" "$code_dir/nar/e16_diagnostics.py" \
    --workdir "$NAR_WORKDIR" \
    --model "${models[$SLURM_ARRAY_TASK_ID]}" \
    --null-space-only \
    --sample-stride 32 \
    --batch-size 1 \
    --output-dir "$output_dir"
