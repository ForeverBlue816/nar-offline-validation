#!/bin/bash
#SBATCH --job-name=nar-e19-finalize
#SBATCH --gpus=pro6000:1
#SBATCH --time=00:20:00
#SBATCH --output=runs/e19-finalize-%j.out
#SBATCH --error=runs/e19-finalize-%j.err
set -euo pipefail
umask 0007

# Idempotent: aggregates whatever row artifacts exist, names the missing ones,
# and depends on no live controller process. Safe to run at any time.
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" PYTHONUNBUFFERED=1 TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"
exec "$python_bin" "$code_dir/nar/e19_qwen3_e2e.py" --workdir "$NAR_WORKDIR" \
    --artifact-root "${NAR_E19_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e19}" \
    --seed "${NAR_E19_SEED:-0}" finalize
