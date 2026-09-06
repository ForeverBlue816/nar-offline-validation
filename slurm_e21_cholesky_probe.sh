#!/bin/bash
#SBATCH --job-name=nar-e21-chol
#SBATCH --gpus=pro6000:1
#SBATCH --time=01:30:00
#SBATCH --mem=90G
#SBATCH --output=runs/e21-chol-%j.out
#SBATCH --error=runs/e21-chol-%j.err
set -euo pipefail
: "${NAR_WORKDIR:?}"; code_dir="${NAR_CODE_DIR:-$PWD}"
export PYTHONPATH="${E13_SITEPACKAGES:-$HOME/.e13_packages}${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
"$NAR_WORKDIR/venv/bin/python" "$code_dir/nar/e21_cholesky_probe.py" \
    "$NAR_WORKDIR/artifacts/e21/llama31_70b/gptq_nar_k8_seed0/cholesky_failure.pt" \
    "$NAR_WORKDIR/results/llama31_70b/e21_cholesky_probe.json"
