#!/bin/bash
set -euo pipefail
E28_WORK=/projects/nar/nar-validation
E28_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PATH="$E28_WORK/e28-env/bin:$PATH"
export PYTHONPATH="$E28_ROOT/quarot-llama3:$E28_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$E28_WORK/cache/huggingface" HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$E28_WORK/cache/xdg" TRITON_CACHE_DIR="$E28_WORK/cache/triton-e28"
export TMPDIR="$E28_WORK/tmp" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$E28_WORK/e28-env/bin/python" -m "e28_v2.$@"
