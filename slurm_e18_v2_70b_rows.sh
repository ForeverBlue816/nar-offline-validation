#!/bin/bash
#SBATCH --job-name=nar-70b-rows
#SBATCH --gpus=pro6000:4
#SBATCH --time=08:00:00
#SBATCH --output=runs/e18v2-70b-rows-%j.out
#SBATCH --error=runs/e18v2-70b-rows-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
out="$NAR_WORKDIR/results/llama31_70b"

# The E18 70B activation-only rows were produced with the v1 bf16 weight fold,
# which the control has since shown destroys the algebra at this scale: the
# Hadamard row reads PPL 15025 against bf16 3.105. Re-measure them on the
# exact-transpose path, which the same control passes on perplexity.
# --report-only because the exact-transpose control clears the 0.01 PPL gate at
# 70B but not the 1e-3 per-chunk gate, that floor being bf16 rounding; the rows
# are therefore diagnostic and are labelled as such rather than certified.
"$python_bin" "$code_dir/nar/e18_v2.py" --workdir "$NAR_WORKDIR" \
    --model-key llama31_70b --eval-sequences 64 --check-rows 32 --report-only --evaluate

for f in e18v2_summary e18v2_per_sequence; do
    [ -f "$out/$f.csv" ] && cp -f "$out/$f.csv" "$out/${f}_exact.csv"
done
echo "== 70B activation-only rows, exact-transpose fold, complete =="
