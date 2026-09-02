#!/bin/bash
#SBATCH --job-name=nar-offline
#SBATCH --gpus=6000ada:1
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#SBATCH --requeue

set -euo pipefail
umask 0007
cd "$(dirname -- "${BASH_SOURCE[0]}")"

echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-}"
nvidia-smi
srun --cpu-bind=cores ./run_all.sh
