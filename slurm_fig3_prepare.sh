#!/bin/bash
#SBATCH --job-name=prism-fig3
#SBATCH --gpus=rtx5090:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/fig3-prepare-%j.out
#SBATCH --error=runs/fig3-prepare-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007

code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg"
export PYTHONUNBUFFERED=1

exec "$python_bin" "$code_dir/figures/prepare_fig3.py" \
    --repo "$code_dir" \
    --workdir "$NAR_WORKDIR" \
    --output-dir "$code_dir/figures" \
    --sequence-batch 4
