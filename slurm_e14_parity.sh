#!/bin/bash
#SBATCH --job-name=nar-e14-parity
#SBATCH --gpus=rtx5090:1
#SBATCH --time=00:10:00
#SBATCH --output=runs/e14-parity-%j.out
#SBATCH --error=runs/e14-parity-%j.err
set -euo pipefail
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${QUAROT_UPSTREAM:?Set QUAROT_UPSTREAM to pinned spcl/QuaRot checkout}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
exec "$NAR_WORKDIR/venv/bin/python" "$code_dir/nar/gptq_parity_check.py" --upstream "$QUAROT_UPSTREAM"
