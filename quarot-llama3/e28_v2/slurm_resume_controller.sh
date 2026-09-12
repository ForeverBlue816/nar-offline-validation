#!/bin/bash
#SBATCH --job-name=e28-v2-controller
#SBATCH --qos=override-limits-but-killable
#SBATCH --constraint=cpu_ok
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=09:00:00
set -euo pipefail
cd "${E28_CODE_ROOT:?Set E28_CODE_ROOT}"
: "${E28_RUN:?Set E28_RUN}"
: "${E28_GPU_ALLOCATION:?Set the already active A40 allocation}"
export E28_REQUIRED_GPU_UUID=GPU-97605b97-fbc5-631d-fd5d-348d2f30ba22
# This batch controller owns no GPU. Its durable srun launches the continuation
# inside the existing GPU allocation, preserving its four assigned CPU cores.
exec env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_GPU srun \
  --jobid="$E28_GPU_ALLOCATION" --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=40960 --cpu-bind=mask_cpu:0xf00000000 \
  bash -c 'set -euo pipefail; python3 -c "import os; assert sorted(os.sched_getaffinity(0)) == [32,33,34,35]"; exec bash quarot-llama3/e28_v2/run.sh pipeline --run "$E28_RUN"' \
  >> "$E28_RUN/slurm-150777-continuation.log" 2>&1
