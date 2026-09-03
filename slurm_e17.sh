#!/bin/bash
#SBATCH --job-name=nar-e17-fused
#SBATCH --gpus=pro6000:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/e17-%j.out
#SBATCH --error=runs/e17-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONUNBUFFERED=1 TMPDIR="$NAR_WORKDIR/tmp" XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg"
mkdir -p "$TMPDIR" "$code_dir/runs"
exec "$python_bin" "$code_dir/nar/e17_fused_r4.py" --workdir "$NAR_WORKDIR"
