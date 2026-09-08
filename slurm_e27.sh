#!/bin/bash
#SBATCH --job-name=nar-e27
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=override-limits-but-killable
#SBATCH --output=runs/e27-%j.out
#SBATCH --error=runs/e27-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TMPDIR="$NAR_WORKDIR/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TMPDIR" "$code_dir/runs"
cd "$code_dir"
"$python_bin" -m unittest discover -s tests -p 'test_e27*.py' -v
"$python_bin" nar/e27_direction_selection.py --workdir "$NAR_WORKDIR" --repo "$code_dir" audit
"$python_bin" nar/e27_direction_selection.py --workdir "$NAR_WORKDIR" --repo "$code_dir" smoke
"$python_bin" nar/e27_direction_selection.py --workdir "$NAR_WORKDIR" --repo "$code_dir" run
