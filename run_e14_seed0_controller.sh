#!/bin/bash
# Monitor the amended six-job E14 seed-0 matrix and submit finalization once.
set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "usage: $0 WORKDIR JOB_3B_HAD JOB_3B_K8 JOB_3B_KMAX JOB_8B_HAD JOB_8B_K8 JOB_8B_KMAX" >&2
    exit 2
fi

workdir="$(readlink -f "$1")"
shift
jobs=("$@")
code_dir="$(cd "$(dirname "$0")" && pwd)"
state_file="$workdir/orchestration/e14_seed0_jobs.tsv"
mkdir -p "$workdir/orchestration" "$code_dir/runs"

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

job_state() {
    local job="$1" output state
    if output="$(squeue -h -j "$job" -o '%T' 2>/dev/null)" &&
        state="$(awk 'NF {print $1; exit}' <<< "$output")" &&
        [[ -n "$state" ]]; then
        printf '%s\n' "$state"
        return 0
    fi
    sacct -X -j "$job" --format=State -n -P |
        awk 'NF {sub(/[+|].*/, "", $1); print $1; exit}'
}

labels=(3b_hadamard 3b_nar_k8 3b_nar_kmax 8b_hadamard 8b_nar_k8 8b_nar_kmax)
: > "$state_file"
for index in "${!jobs[@]}"; do
    printf '%s\t%s\n' "${labels[$index]}" "${jobs[$index]}" >> "$state_file"
done

while true; do
    remaining=0
    for index in "${!jobs[@]}"; do
        state="$(job_state "${jobs[$index]}")"
        case "$state" in
            COMPLETED) ;;
            PENDING|RUNNING|REQUEUED|REQUEUE_FED|CONFIGURING|COMPLETING|RESIZING|SUSPENDED)
                remaining=$((remaining + 1))
                ;;
            *)
                log "STOP: ${labels[$index]} job=${jobs[$index]} state=${state:-UNKNOWN}"
                exit 1
                ;;
        esac
    done
    log "seed0 matrix remaining=$remaining/6"
    [[ "$remaining" -eq 0 ]] && break
    sleep 60
done

if [[ -f "$workdir/results/E14_DONE.json" ]]; then
    log "E14 final output already exists"
    exit 0
fi
finalize="$(sbatch --parsable --qos=override-limits-but-killable --export="ALL,NAR_WORKDIR=$workdir,NAR_CODE_DIR=$code_dir" "$code_dir/slurm_e14_finalize.sh")"
finalize="${finalize%%;*}"
printf 'finalize\t%s\n' "$finalize" >> "$state_file"
log "finalize job=$finalize"

while true; do
    state="$(job_state "$finalize")"
    case "$state" in
        COMPLETED)
            log "E14 seed-0 matrix complete"
            exit 0
            ;;
        PENDING|RUNNING|CONFIGURING|COMPLETING) sleep 30 ;;
        *)
            log "STOP: finalize job=$finalize state=${state:-UNKNOWN}"
            exit 1
            ;;
    esac
done
