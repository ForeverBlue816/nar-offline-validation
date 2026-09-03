#!/bin/bash
#SBATCH --job-name=nar-e15-audit
#SBATCH --gpus=rtx5090:1
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/e15-followup-%j.out
#SBATCH --error=runs/e15-followup-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONUNBUFFERED=1 TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$NAR_WORKDIR/runs" "$TMPDIR" "$code_dir/runs"
exec "$python_bin" "$code_dir/nar/e15_followup.py" --workdir "$NAR_WORKDIR"
