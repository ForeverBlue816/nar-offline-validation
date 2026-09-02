#!/bin/bash
# Fail-closed Slurm continuation for the already-running E13 through E15.
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 WORKDIR E13_3B_JOB E13_8B_JOB E13_REPORT_JOB PARITY_JOB E14_3B_CAL_JOB" >&2
    exit 2
fi

workdir="$1"
e13_3b="$2"
e13_8b="$3"
e13_report="$4"
parity="$5"
cal_3b="$6"
code_dir="$(cd "$(dirname "$0")" && pwd)"
state_dir="$workdir/orchestration"
state_file="$state_dir/e13_e15_jobs.tsv"
mkdir -p "$state_dir"

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
            *)
                sleep 60
                ;;
        esac
    done
}

wait_submit_slot() {
    while [[ "$(squeue -h -u "$(id -un)" | wc -l)" -ge 5 ]]; do
        sleep 60
    done
}

submit() {
    local output
    output="$(sbatch --parsable "$@")"
    printf '%s' "${output%%;*}"
}

record e13_3b "$e13_3b"
record e13_8b "$e13_8b"
record e13_report "$e13_report"
record gptq_parity "$parity"
record e14_cal_3b "$cal_3b"

wait_job e13_3b "$e13_3b"
wait_submit_slot
cal_8b="$(submit --dependency="afterok:$e13_report:$parity" \
    --export="ALL,NAR_WORKDIR=$workdir,NAR_MODEL=llama31_8b,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e14_calibrate.sh")"
record e14_cal_8b "$cal_8b"

wait_job e13_8b "$e13_8b"
wait_job e13_report "$e13_report"
wait_job gptq_parity "$parity"
wait_job e14_cal_3b "$cal_3b"
wait_job e14_cal_8b "$cal_8b"

wait_submit_slot
pipe_3b="$(submit --export="ALL,NAR_WORKDIR=$workdir,NAR_MODEL=llama32_3b,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e14_pipeline.sh")"
record e14_pipeline_3b "$pipe_3b"
wait_submit_slot
pipe_8b="$(submit --export="ALL,NAR_WORKDIR=$workdir,NAR_MODEL=llama31_8b,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e14_pipeline.sh")"
record e14_pipeline_8b "$pipe_8b"
wait_job e14_pipeline_3b "$pipe_3b"
wait_job e14_pipeline_8b "$pipe_8b"

e14_report_job="$(submit --export="ALL,NAR_WORKDIR=$workdir,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e14_finalize.sh")"
record e14_report "$e14_report_job"
wait_job e14_report "$e14_report_job"

e15_job="$(submit --export="ALL,NAR_WORKDIR=$workdir,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e15.sh")"
record e15 "$e15_job"
wait_job e15 "$e15_job"

e15_report_job="$(submit --export="ALL,NAR_WORKDIR=$workdir,NAR_CODE_DIR=$code_dir" \
    "$code_dir/slurm_e15_report.sh")"
record e15_report "$e15_report_job"
wait_job e15_report "$e15_report_job"
log "E13-E15 continuation complete"
