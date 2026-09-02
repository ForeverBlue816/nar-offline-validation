#!/bin/bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$project_dir/cache/huggingface"
export HF_DATASETS_CACHE="$project_dir/cache/datasets"
export XDG_CACHE_HOME="$project_dir/cache/xdg"
export MPLCONFIGDIR="$project_dir/cache/matplotlib"
export TMPDIR="$project_dir/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python_bin="${NAR_PYTHON:-$project_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" && -x "$project_dir/venv/bin/python" ]]; then
    python_bin="$project_dir/venv/bin/python"
fi

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR" "$project_dir/runs"
exec "$python_bin" "$project_dir/nar/experiment.py" --workdir "$project_dir" all --batch-size 2 --eval-sequences 64
