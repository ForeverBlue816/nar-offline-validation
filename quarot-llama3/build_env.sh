#!/bin/bash
# E28: build QuaRot's CUDA extension and fast-hadamard-transform into the E28 venv.
#
#   NAR_WORKDIR=/projects/_hdd/nar/nar-validation bash quarot-llama3/build_env.sh
#
# The venv ($NAR_WORKDIR/e28-env) holds QuaRot's pinned stack (torch 2.4.1+cu124,
# transformers 4.36.2, flash-attn 2.6.3 wheel).  nvcc comes from the cluster's
# CUDA/12.4.0 environment module; QuaRot's own setup.py chooses the gencode
# list (sm_75/80/86), fast-hadamard-transform's chooses sm_70/80(/90).
set -euo pipefail
W="${NAR_WORKDIR:?Set NAR_WORKDIR}"
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load CUDA/12.4.0
export CUDA_HOME="${EBROOTCUDA:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export PATH="$W/e28-env/bin:$CUDA_HOME/bin:$PATH"
export TMPDIR="$W/tmp/pip-tmp" PIP_CACHE_DIR="$W/tmp/pip-cache" MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
nvcc --version | tail -2
cd "$W/external/quarot"
git rev-parse HEAD
# 1. fast-hadamard-transform (submodule, v1.0.4.post1) — QuaRot's fused Hadamard.
pip install --no-build-isolation -v third-party/fast-hadamard-transform 2>&1 | tail -5
# 2. quarot._CUDA: CUTLASS INT4 GEMM, sym_quant/sym_dequant, FlashInfer INT4 KV cache
#    (with the GQA decode patch applied to the checkout).
pip install --no-build-isolation -v --no-deps -e . 2>&1 | tail -5
python -c "import quarot, quarot._CUDA, fast_hadamard_transform; print('quarot ok', quarot.__file__)"
