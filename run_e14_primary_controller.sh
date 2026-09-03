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
accept_failed_anchor="${NAR_ACCEPT_FAILED_ANCHOR:-0}"
resume_controller="${NAR_RESUME_CONTROLLER:-0}"
mkdir -p "$state_dir" "$code_dir/runs"

if [[ -s "$state_file" && "$resume_controller" != 1 ]]; then
    echo "refusing duplicate E14 submission; manifest exists: $state_file" >&2
    exit 2
fi
if [[ "$resume_controller" != 1 ]]; then
    : > "$state_file"
fi

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

record() {
    printf '%s\t%s\n' "$1" "$2" >> "$state_file"
    log "$1 job=$2"
}

job_state() {
    local output state
    for attempt in 1 2 3 4 5; do
        if output="$(squeue -h -j "$1" -o '%T')" &&
            state="$(awk 'NF {print $1; exit}' <<< "$output")" &&
            [[ -n "$state" ]]; then
            printf '%s\n' "$state"
            return 0
        fi
        if output="$(sacct -X -j "$1" --format=State -n -P)"; then
            awk 'NF {sub(/[+|].*/, "", $1); print $1; exit}' <<< "$output"
            return 0
        fi
        printf '%s WARN sacct query failed job=%s attempt=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$attempt" >&2
        sleep 10
    done
    return 1
}

require_success() {
    local label="$1" job="$2" state
    if ! state="$(job_state "$job")"; then
        log "WARN: unable to query $label job=$job; retaining it"
        return 0
    fi
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
        if ! state="$(job_state "$job")"; then
            log "WARN: unable to query $label job=$job; waiting"
            sleep 30
            continue
        fi
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

wait_anchor() {
    local label="$1"
    local job_id="$2"

    while true; do
        local state
        if ! state="$(job_state "$job_id")"; then
            log "WARN: unable to query $label job=$job_id; waiting"
            sleep 30
            continue
        fi
        case "$state" in
            COMPLETED)
                log "$label job=$job_id state=$state"
                return 0
                ;;
            FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
                if [[ "$accept_failed_anchor" == "1" ]]; then
                    log "OVERRIDE: $label job=$job_id state=$state; proceeding with recorded anchor failure"
                    printf 'anchor_override\taccepted_failed_anchor:%s\n' "$job_id" >> "$state_file"
                    return 0
                fi
                log "STOP: $label job=$job_id state=$state"
                return 1
                ;;
            *)
                log "$label job=$job_id state=${state:-UNKNOWN}; waiting"
                sleep 30
                ;;
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

if [[ "$resume_controller" == 1 ]]; then
    recorded_anchor="$(awk -F '\t' '$1 == "anchor_gate" {print $2; exit}' "$state_file")"
    if [[ "$recorded_anchor" != "$anchor_job" ]]; then
        log "STOP: resume anchor mismatch manifest=$recorded_anchor requested=$anchor_job"
        exit 2
    fi
    if awk -F '\t' '$1 == "anchor_override" {found=1} END {exit !found}' "$state_file"; then
        log "RESUME: retaining recorded failed-anchor override job=$anchor_job"
    else
        wait_anchor anchor_gate "$anchor_job"
    fi
else
    record anchor_gate "$anchor_job"
    wait_anchor anchor_gate "$anchor_job"
fi

jobs=()
for ((task = 0; task < 18; task++)); do
    label="primary_task_$task"
    if [[ "$resume_controller" == 1 ]]; then
        recorded_job="$(awk -F '\t' -v label="$label" '$1 == label {print $2; exit}' "$state_file")"
        if [[ -n "$recorded_job" ]]; then
            jobs[$task]="$recorded_job"
            log "RESUME: $label job=$recorded_job"
            continue
        fi
    fi
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
    jobs[$task]="$job"
    record "$label" "$job"
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
