#!/bin/bash
#SBATCH --job-name=nar-e17-v2
#SBATCH --gpus=pro6000:1
#SBATCH --qos=override-limits-but-killable
#SBATCH --time=01:30:00
#SBATCH --output=runs/e17-v2-%j.out
#SBATCH --error=runs/e17-v2-%j.err
set -euo pipefail
umask 0007

code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONUNBUFFERED=1 TMPDIR="$NAR_WORKDIR/tmp" XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg"
mkdir -p "$TMPDIR" "$code_dir/runs"
exec "$python_bin" "$code_dir/nar/e17_v2.py" --workdir "$NAR_WORKDIR" "$@"
