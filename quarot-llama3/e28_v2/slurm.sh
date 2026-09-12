#!/bin/bash
#SBATCH --job-name=nar-e28-v2
#SBATCH --gpus=a40:1
#SBATCH --qos=rose
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -euo pipefail
E28_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# sbatch copies this script, so use its explicit submit directory.
E28_ROOT=${E28_CODE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$E28_ROOT"
: "${E28_RUN:?Set E28_RUN}"
run() { bash "$E28_ROOT/quarot-llama3/e28_v2/run.sh" "$@" --run "$E28_RUN"; }
run audit
run verify --level operators
for model in 3b 8b; do run verify --level r4 --model "$model"; done
for model in 3b 8b; do
  for method in fp16 hadamard nar; do run verify --level model --model "$model" --method "$method"; done
done
touch "$E28_RUN/verification.done"
run pipeline

if [[ ${E28_STREAM_GRAPH:-0} == 1 ]]; then run stream_pipeline; fi
