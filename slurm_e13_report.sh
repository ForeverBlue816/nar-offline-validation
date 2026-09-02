#!/bin/bash
#SBATCH --job-name=nar-e13-report
#SBATCH --time=00:10:00
#SBATCH --output=runs/e13-report-%j.out
#SBATCH --error=runs/e13-report-%j.err
set -euo pipefail
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
exec "$NAR_WORKDIR/venv/bin/python" "$code_dir/nar/e13_report.py" --workdir "$NAR_WORKDIR"
