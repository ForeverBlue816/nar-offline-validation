#!/bin/bash
#SBATCH --job-name=nar-e22-probe
#SBATCH --gpus=a40:1
#SBATCH --time=00:40:00
#SBATCH --output=runs/e22-probe-%j.out
#SBATCH --error=runs/e22-probe-%j.err
set -euo pipefail
: "${NAR_WORKDIR:?}"; code_dir="${NAR_CODE_DIR:-$PWD}"
export PYTHONPATH="${E13_SITEPACKAGES:-$HOME/.e13_packages}${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$NAR_WORKDIR/venv/bin/python" "$code_dir/nar/e22_batch_probe.py" --workdir "$NAR_WORKDIR" --model "${NAR_E22_MODEL:-qwen3_0.6b_base}"
