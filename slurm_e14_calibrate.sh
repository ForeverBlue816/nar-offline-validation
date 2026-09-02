#!/bin/bash
#SBATCH --job-name=nar-e14-cal
#SBATCH --gpus=rtx5090:1
#SBATCH --time=08:00:00
#SBATCH --output=runs/e14-cal-%j.out
#SBATCH --error=runs/e14-cal-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
: "${NAR_MODEL:?Set NAR_MODEL to llama32_3b or llama31_8b}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec "$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" calibrate --model "$NAR_MODEL"
