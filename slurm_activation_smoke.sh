#!/bin/bash
#SBATCH --job-name=nar-act-smoke
#SBATCH --gpus=rtx5090:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/activation-smoke-%j.out
#SBATCH --error=runs/activation-smoke-%j.err

set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR to project storage}"
work_dir="$NAR_WORKDIR"
smoke="$work_dir/smoke_activation"
python_bin="$work_dir/venv/bin/python"
export HF_HOME="$work_dir/cache/huggingface"
export HF_DATASETS_CACHE="$work_dir/cache/datasets"
export XDG_CACHE_HOME="$work_dir/cache/xdg"
export MPLCONFIGDIR="$work_dir/cache/matplotlib"
export TMPDIR="$work_dir/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$work_dir/runs" "$work_dir/tmp" "$smoke/results" "$smoke/runs" "$smoke/activations/llama32_3b"
for shared in cache venv; do
  if [ ! -e "$smoke/$shared" ]; then ln -s "$work_dir/$shared" "$smoke/$shared"; fi
done
"$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$work_dir" calibrate --model llama32_3b
for shared in wide_cal_a activation_factors; do
  if [ ! -e "$smoke/activations/llama32_3b/$shared" ]; then
    ln -s "$work_dir/activations/llama32_3b/$shared" "$smoke/activations/llama32_3b/$shared"
  fi
done
"$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$smoke" e5 --model llama32_3b --eval-sequences 1 --seq-len 128 --seeds 1
"$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$smoke" e6 --verify-rows 2 --dense-row-batch 512 --benchmark-tokens 1 --warmup 1 --repeats 2
"$python_bin" "$code_dir/nar/activation_diagnostics.py" --workdir "$smoke" collect-v --sequences 2 --seq-len 128 --sample-stride 32 --batch-size 1 --checkpoint-sequences 1
"$python_bin" "$code_dir/nar/activation_diagnostics.py" --workdir "$smoke" e7 --max-layers 1
"$python_bin" "$code_dir/nar/activation_diagnostics.py" --workdir "$smoke" collect-down-heldout --sequences 2 --seq-len 128 --sample-stride 32 --batch-size 1 --checkpoint-sequences 1
"$python_bin" "$code_dir/nar/activation_diagnostics.py" --workdir "$smoke" e8 --steps 2 --batch-size 4 --metric-batch 64 --max-layers 1
"$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$smoke" calibrate --model llama32_1b --calibration-sequences 2 --seq-len 128 --batch-size 1
"$python_bin" "$code_dir/nar/activation_experiments.py" --workdir "$smoke" e5 --model llama32_1b --eval-sequences 1 --seq-len 128 --seeds 1
