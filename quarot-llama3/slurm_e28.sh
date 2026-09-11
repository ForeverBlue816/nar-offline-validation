#!/bin/bash
#SBATCH --job-name=nar-e28
#SBATCH --gpus=a40:1
#SBATCH --qos=rose
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=runs/e28-%j.out
#SBATCH --error=runs/e28-%j.err
# E28 end-to-end chain on one A40 (sm_86): three rows x two models, then the
# E17 v3 kernel microbenchmark on the same GPU.  Also runnable inside an
# existing allocation:  srun --jobid=<id> --overlap bash quarot-llama3/slurm_e28.sh
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
export NAR_CODE_DIR="$code_dir"
cd "$code_dir"
run() { bash "$code_dir/quarot-llama3/run_e28.sh" python "$code_dir/quarot-llama3/e28_bench.py" "$@"; }
models="${NAR_E28_MODELS:-3b 8b}"
rows="${NAR_E28_ROWS:-fp16 hadamard nar}"
for model in $models; do
  [ -f "$code_dir/results/e28/$model/nar_kernel_selection.json" ] || run select --model "$model"
  for row in $rows; do
    run bench --model "$model" --row "$row"
  done
done
run collect --models $models

# E17 v3 kernel microbenchmark on the A40 (main repository environment; the
# scratch workdir symlinks activations/artifacts so nothing outside results/e28
# is written).
if [ "${NAR_E28_E17:-1}" = 1 ]; then
  scratch="$NAR_WORKDIR/tmp/e28/e17_workdir"
  mkdir -p "$scratch/results" "$scratch/runs"
  ln -sfn "$NAR_WORKDIR/activations" "$scratch/activations"
  ln -sfn "$NAR_WORKDIR/artifacts" "$scratch/artifacts"
  ln -sfn "$NAR_WORKDIR/cache" "$scratch/cache"
  export HF_HOME="$NAR_WORKDIR/cache/huggingface" XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg"
  export TRITON_CACHE_DIR="$NAR_WORKDIR/cache/triton-e28-main" TMPDIR="$NAR_WORKDIR/tmp" PYTHONUNBUFFERED=1
  mkdir -p "$TRITON_CACHE_DIR"
  "$NAR_WORKDIR/venv/bin/python" "$code_dir/nar/e17_v3.py" --workdir "$scratch" \
    --models llama32_3b llama31_8b --ranks 8 --tokens 1 32 2048
  for model in llama32_3b llama31_8b; do
    short=$([ "$model" = llama32_3b ] && echo 3b || echo 8b)
    mkdir -p "$code_dir/results/e28/$short"
    for f in e17v3_fused_r4_timings.csv e17v3_kernel_a_backends.csv e17v3_verification.json; do
      cp "$scratch/results/$model/$f" "$code_dir/results/e28/$short/a40_$f"
    done
  done
  cp "$scratch/results/llama32_3b/E17V3_DONE.json" "$code_dir/results/e28/a40_E17V3_DONE.json"
fi
echo "E28 chain done"
