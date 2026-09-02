#!/bin/bash
#SBATCH --job-name=nar-e15-fp4
#SBATCH --gpus=rtx5090:1
#SBATCH --time=04:00:00
#SBATCH --output=runs/e15-%j.out
#SBATCH --error=runs/e15-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
exec "$NAR_WORKDIR/venv/bin/python" "$code_dir/nar/e15_fp4.py" --workdir "$NAR_WORKDIR"
