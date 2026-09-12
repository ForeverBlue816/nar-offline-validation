#!/bin/bash
#SBATCH --job-name=e28-stream-build
#SBATCH --qos=rose
#SBATCH --constraint=cpu_ok
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
set -euo pipefail
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh
module load CUDA/12.4.0
export CUDA_HOME="${EBROOTCUDA}"
export MAX_JOBS=2
cd "${E28_CODE_ROOT:?Set E28_CODE_ROOT}"
bash quarot-llama3/e28_v2/run.sh stream_backend --run "${E28_RUN:?Set E28_RUN}"
