#!/bin/bash
#SBATCH --job-name=nar-e14-gptq
#SBATCH --gpus=rtx5090:1
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/e14-gptq-%j.out
#SBATCH --error=runs/e14-gptq-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
: "${NAR_MODEL:?Set NAR_MODEL}"
: "${NAR_ROTATION:?Set NAR_ROTATION to hadamard or nar}"
: "${NAR_SEED:?Set NAR_SEED explicitly}"
artifact_root="${NAR_E14_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e14}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec "$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
    --artifact-root "$artifact_root" --seed "$NAR_SEED" gptq \
    --model "$NAR_MODEL" --rotation "$NAR_ROTATION"
