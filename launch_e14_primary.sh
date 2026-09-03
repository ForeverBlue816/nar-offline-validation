#!/bin/bash
# Queue the frozen E14 primary matrix behind a successful QuaRot anchor gate.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 WORKDIR ANCHOR_GATE_JOB" >&2
    exit 2
fi

workdir="$(readlink -f "$1")"
anchor_job="$2"
code_dir="$(cd "$(dirname "$0")" && pwd)"
state_dir="$workdir/orchestration"
state_file="$state_dir/e14_primary_jobs.tsv"
mkdir -p "$state_dir" "$code_dir/runs"

if [[ -s "$state_file" && "${NAR_ALLOW_RESUBMIT:-0}" != 1 ]]; then
    echo "refusing duplicate E14 submission; manifest exists: $state_file" >&2
    exit 2
fi
: > "$state_file"

submit() {
    local output
    if ! output="$(sbatch --parsable "$@")"; then
        return 1
    fi
    output="${output%%;*}"
    [[ -n "$output" ]] || return 1
    printf '%s' "$output"
}

record() {
    printf '%s\t%s\n' "$1" "$2" >> "$state_file"
    printf '%s job=%s\n' "$1" "$2"
}

matrix="$(submit --dependency="afterok:$anchor_job" --array=0-17%4 \
    --export="ALL,NAR_WORKDIR=$workdir,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e14_primary_array.sh")"
record primary_matrix "$matrix"

finalize="$(submit --dependency="afterok:$matrix" \
    --export="ALL,NAR_WORKDIR=$workdir,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e14_finalize.sh")"
record finalize "$finalize"
record anchor_gate "$anchor_job"
