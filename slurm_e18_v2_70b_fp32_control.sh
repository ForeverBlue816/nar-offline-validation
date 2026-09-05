#!/bin/bash
#SBATCH --job-name=nar-70b-fp32
#SBATCH --gpus=pro6000:8
#SBATCH --time=08:00:00
#SBATCH --output=runs/e18v2-70b-fp32-%j.out
#SBATCH --error=runs/e18v2-70b-fp32-%j.err
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

# Gate on the end-to-end 70B. The bf16 exact-transpose control clears the 0.01
# PPL gate but sits at 2.4-3.2e-3 on the per-chunk gate of 1e-3, which is bf16
# rounding rather than an algebra error. fp32 containers hold the same weights
# and remove that floor; if the control still fails here, the 70B algebra
# cannot be certified and the end-to-end run should not be started.
# 263 GiB of fp32 weights over 8 GPUs is 33 GiB each, well inside the budget.
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader
"$python_bin" "$code_dir/nar/e18_v2.py" --workdir "$NAR_WORKDIR" \
    --model-key llama31_70b --eval-sequences 64 --check-rows 32 \
    --compute-dtype float32 --report-only --rotation-only-control

cp -f "$out/e18v2_rotation_only_control.csv" "$out/e18v2_rotation_only_control_exact_fp32.csv"
cp -f "$out/e18v2_fold_audit.json" "$out/e18v2_fold_audit_exact_fp32.json"
echo "== 70B fp32 rotation-only control complete =="
"$python_bin" - "$out/e18v2_rotation_only_control_exact_fp32.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    print(f"{r['method']:9s} dPPL={float(r['ppl_abs_difference']):.2e} "
          f"max|dNLL|={float(r['max_abs_nll_delta']):.2e} passed={r['passed']}")
print("GATE:", "PASS - end-to-end 70B is worth starting"
      if rows and all(r["passed"] == "True" for r in rows)
      else "FAIL - do not start end-to-end 70B")
PY
