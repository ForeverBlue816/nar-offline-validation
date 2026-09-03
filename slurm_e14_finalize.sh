#!/bin/bash
#SBATCH --job-name=nar-e14-final
#SBATCH --time=00:10:00
#SBATCH --output=runs/e14-final-%j.out
#SBATCH --error=runs/e14-final-%j.err
set -euo pipefail
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
"$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" finalize
"$python_bin" "$code_dir/nar/e14_report.py" --workdir "$NAR_WORKDIR"
"$python_bin" "$code_dir/nar/publish_results.py" "$NAR_WORKDIR/results" "$code_dir/results"
cp "$NAR_WORKDIR/report.md" "$code_dir/report.md"
