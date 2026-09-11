#!/bin/bash
#SBATCH --job-name=nar-hf-upload
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/hf-upload-%j.out
#SBATCH --error=runs/hf-upload-%j.out
#SBATCH --requeue
# CPU-only archive job: waits for the lock if a session-side uploader holds it,
# then runs the resumable upload sequence (tmp/hf_upload_all.sh).
set -euo pipefail
W=/projects/_hdd/nar/nar-validation
cd "${SLURM_SUBMIT_DIR:-/home/yanlongc/nar-offline-validation}"
exec flock "$W/tmp/hf_upload.lock" "$W/tmp/hf_upload_all.sh" >> runs/hf-upload.log 2>&1
