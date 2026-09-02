#!/bin/bash
#SBATCH --job-name=nar-e12-wy
#SBATCH --gpus=rtx5090:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/e12-%j.out
#SBATCH --error=runs/e12-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to project storage}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONUNBUFFERED=1
exec "$python_bin" "$code_dir/nar/e12_wy.py" --workdir "$NAR_WORKDIR"
