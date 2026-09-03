#!/bin/bash
#SBATCH --job-name=nar-e17-v3
#SBATCH --gpus=pro6000:1
#SBATCH --qos=override-limits-but-killable
#SBATCH --time=08:00:00
#SBATCH --output=runs/e17-v3-%j.out
#SBATCH --error=runs/e17-v3-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg"
export PYTHONUNBUFFERED=1 TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
exec "$python_bin" "$code_dir/nar/e17_v3.py" --workdir "$NAR_WORKDIR" "$@"
