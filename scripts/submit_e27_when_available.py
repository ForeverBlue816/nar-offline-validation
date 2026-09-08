#!/usr/bin/env python3
"""Wait for an existing permitted Slurm submission slot; never cancel another job."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--workdir', type=Path, required=True)
    args = p.parse_args()
    status = args.repo / 'experiments/e27_execution.json'
    lock_path = args.workdir / 'orchestration/e27_submission.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(status.read_text()) if status.exists() else {}
        if previous.get('job_id'):
            print('E27 already submitted:', previous['job_id'], flush=True)
            return
        env = {**os.environ, 'NAR_WORKDIR': str(args.workdir), 'NAR_CODE_DIR': str(args.repo)}
        attempts = 0
        while True:
            attempts += 1
            result = subprocess.run(['sbatch', '--parsable', 'slurm_e27.sh'], cwd=args.repo,
                                    env=env, capture_output=True, text=True)
            record = {'experiment': 'E27', 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                      'attempts': attempts, 'requested_gpu': 'rtx5090:1', 'models_sequential': ['llama32_3b', 'llama31_8b'],
                      'qos': 'override-limits-but-killable', 'preregistration': 'experiments/e27_preregistration.json'}
            if result.returncode == 0:
                job_id = result.stdout.strip().split(';')[0]
                if not job_id.isdigit():
                    raise RuntimeError(f'Unrecognized sbatch response {result.stdout}')
                record.update(status='submitted', job_id=job_id)
            else:
                retry = any(x in result.stderr for x in ['QOSMaxSubmitJobPerUserLimit', 'AssocMaxSubmitJobLimit', 'Socket timed out', 'temporarily unavailable'])
                record.update(status='waiting_for_submission_slot' if retry else 'submission_failed',
                              last_error=result.stderr.strip())
            temp = status.with_suffix('.json.tmp')
            temp.write_text(json.dumps(record, indent=2) + '\n'); os.replace(temp, status)
            print(json.dumps(record), flush=True)
            if result.returncode == 0:
                return
            if not retry:
                raise SystemExit(result.returncode)
            time.sleep(60)


if __name__ == '__main__':
    main()
