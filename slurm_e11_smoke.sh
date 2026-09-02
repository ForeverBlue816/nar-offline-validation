#!/bin/bash
#SBATCH --job-name=nar-e11-smoke
#SBATCH --gpus=rtx5090:1
#SBATCH --time=01:00:00
#SBATCH --output=runs/e11-smoke-%j.out
#SBATCH --error=runs/e11-smoke-%j.err
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to project storage}"
python_bin="$NAR_WORKDIR/venv/bin/python"
smoke_dir="$NAR_WORKDIR/smoke_e11_stream"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" MPLCONFIGDIR="$NAR_WORKDIR/cache/matplotlib" TMPDIR="$NAR_WORKDIR/tmp"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$smoke_dir" "$TMPDIR" "$MPLCONFIGDIR"
"$python_bin" "$code_dir/nar/e11_fair_baselines.py" --workdir "$smoke_dir" --asset-workdir "$NAR_WORKDIR" calibrate --model llama32_3b --calibration-sequences 2 --seq-len 128 --batch-size 1 --sample-stride 32 --max-layers 1
exec "$python_bin" "$code_dir/nar/e11_fair_baselines.py" --workdir "$smoke_dir" --asset-workdir "$NAR_WORKDIR" evaluate --model llama32_3b --eval-sequences 1 --seq-len 128 --seeds 1 --max-layers 1
