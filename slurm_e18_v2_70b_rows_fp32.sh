#!/bin/bash
#SBATCH --job-name=nar-70b-rows32
#SBATCH --gpus=pro6000:8
#SBATCH --time=10:00:00
#SBATCH --output=runs/e18v2-70b-rows32-%j.out
#SBATCH --error=runs/e18v2-70b-rows32-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
out="$NAR_WORKDIR/results/llama31_70b"

# The bf16 rows survive the exact-transpose fold unchanged - Hadamard reads
# 15137 against 15025 under the v1 weight fold - so the fold is not the cause.
# What did change with fp32 containers is the control: in bf16 it sat at
# 2.4e-3 on the per-chunk gate of 1e-3, in fp32 it passes at 1e-6 to 7e-5.
# That is a three-order-of-magnitude noise floor difference, so the rows are
# re-measured on the same fp32 path before the Hadamard result is believed.
mv -f "$out/e18v2_summary.csv" "$out/e18v2_summary_bf16.csv"
mv -f "$out/e18v2_per_sequence.csv" "$out/e18v2_per_sequence_bf16.csv"

"$NAR_WORKDIR/venv/bin/python" "$code_dir/nar/e18_v2.py" --workdir "$NAR_WORKDIR" \
    --model-key llama31_70b --eval-sequences 64 --check-rows 32 \
    --compute-dtype float32 --report-only --evaluate

cp -f "$out/e18v2_summary.csv" "$out/e18v2_summary_fp32.csv"
cp -f "$out/e18v2_per_sequence.csv" "$out/e18v2_per_sequence_fp32.csv"
"$NAR_WORKDIR/venv/bin/python" - "$out/e18v2_summary_fp32.csv" "$out/e18v2_summary_bf16.csv" <<'PY'
import csv, sys
def rows(p): return {r["method"]: float(r["ppl"]) for r in csv.DictReader(open(p))}
a, b = rows(sys.argv[1]), rows(sys.argv[2])
print(f"{'method':10s}{'fp32':>16s}{'bf16':>16s}")
for m in a:
    print(f"{m:10s}{a[m]:16.5f}{b.get(m, float('nan')):16.5f}")
PY
