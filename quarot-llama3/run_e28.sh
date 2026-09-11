#!/bin/bash
# E28 environment wrapper: run a command inside the E28 venv on a GPU node.
#   NAR_WORKDIR=... bash quarot-llama3/run_e28.sh python quarot-llama3/e28_bench.py bench --model 3b --row hadamard
set -euo pipefail
W="${NAR_WORKDIR:?Set NAR_WORKDIR}"
code_dir="${NAR_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PATH="$W/e28-env/bin:$PATH"
export HF_HOME="$W/cache/huggingface" XDG_CACHE_HOME="$W/cache/xdg" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRITON_CACHE_DIR="$W/cache/triton-e28" TMPDIR="$W/tmp" PYTHONUNBUFFERED=1
export PYTHONPATH="$code_dir/quarot-llama3:$code_dir${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
cd "$code_dir"
exec "$@"
