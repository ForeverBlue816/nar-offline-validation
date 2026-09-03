#!/bin/bash
# QOS-aware controller for the frozen E14 primary matrix.
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
max_submitted=4
mkdir -p "$state_dir" "$code_dir/runs"

if [[ -s "$state_file" && "${NAR_ALLOW_RESUBMIT:-0}" != 1 ]]; then
    echo "refusing duplicate E14 submission; manifest exists: $state_file" >&2
    exit 2
fi
: > "$state_file"

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

record() {
    printf '%s\t%s\n' "$1" "$2" >> "$state_file"
    log "$1 job=$2"
}

job_state() {
    sacct -X -j "$1" --format=State -n -P | awk 'NF {sub(/[+|].*/, "", $1); print $1; exit}'
}

require_success() {
    local label="$1" job="$2" state
    state="$(job_state "$job")"
    case "$state" in
        FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE)
            log "STOP: $label job=$job state=$state"
            exit 1
            ;;
    esac
}

wait_job() {
    local label="$1" job="$2" state
    while true; do
        state="$(job_state "$job")"
        case "$state" in
            COMPLETED)
                log "$label completed job=$job"
                return
                ;;
            FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE)
                log "STOP: $label job=$job state=$state"
                exit 1
                ;;
            *) sleep 60 ;;
        esac
    done
}

submit() {
    local output
    if ! output="$(sbatch --parsable "$@")"; then
        return 1
    fi
    output="${output%%;*}"
    [[ -n "$output" ]] || return 1
    printf '%s' "$output"
}

record anchor_gate "$anchor_job"
wait_job anchor_gate "$anchor_job"

jobs=()
for ((task = 0; task < 18; task++)); do
    for prior in "${jobs[@]}"; do
        require_success "primary_task" "$prior"
    done
    while [[ "$(squeue -h -u "$(id -un)" | wc -l)" -ge "$max_submitted" ]]; do
        sleep 60
        for prior in "${jobs[@]}"; do
            require_success "primary_task" "$prior"
        done
    done
    job="$(submit --array="$task" \
        --export="ALL,NAR_WORKDIR=$workdir,NAR_CODE_DIR=$code_dir" \
        "$code_dir/slurm_e14_primary_array.sh")"
    jobs+=("$job")
    record "primary_task_$task" "$job"
done

for index in "${!jobs[@]}"; do
    wait_job "primary_task_$index" "${jobs[$index]}"
done

while [[ "$(squeue -h -u "$(id -un)" | wc -l)" -ge "$max_submitted" ]]; do
    sleep 60
done
finalize="$(submit --export="ALL,NAR_WORKDIR=$workdir,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e14_finalize.sh")"
record finalize "$finalize"
wait_job finalize "$finalize"
log "E14 primary matrix complete"
