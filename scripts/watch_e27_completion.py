#!/usr/bin/env python3
"""Monitor only E27, verify completed artifacts, and publish its authorized outputs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def run(command, repo, check=True):
    return subprocess.run(command, cwd=repo, capture_output=True, text=True, check=check)


def write_status(path, record):
    record['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n'); os.replace(temp, path)


def without_e27(text):
    text = re.sub(r'<!-- E27_RESULTS_BEGIN -->.*?<!-- E27_RESULTS_END -->', '', text, flags=re.S)
    text = re.sub(r'^## E27 — which tokens define the subspace\n.*?(?=^#{1,2} |\Z)', '', text, flags=re.M | re.S)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    status = args.repo / 'experiments/e27_execution.json'
    lock_path = args.workdir / 'orchestration/e27_completion.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            record = json.loads(status.read_text()) if status.exists() else {}
            if record.get('job_id'):
                break
            if record.get('status') == 'submission_failed':
                raise RuntimeError(record.get('last_error'))
            time.sleep(60)
        job = record['job_id']
        while True:
            query = run(['sacct', '-j', job, '--noheader', '--parsable2', '--format=JobIDRaw,State,ExitCode'], args.repo, check=False)
            matches = [line.split('|') for line in query.stdout.splitlines() if line.split('|')[0] == job]
            state = matches[0][1].split()[0] if matches else 'UNKNOWN'
            record.update(status='running' if state == 'RUNNING' else 'queued' if state in ('PENDING', 'CONFIGURING', 'REQUEUED') else state.lower(), scheduler_state=state)
            write_status(status, record)
            if state == 'COMPLETED':
                break
            if state in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE'):
                record['status'] = 'experiment_failed'
                record['error_log'] = f'runs/e27-{job}.err'
                write_status(status, record)
                raise RuntimeError(f'E27 {job} {state}; inspect {record["error_log"]}')
            time.sleep(60)
        expected = hashlib.sha256((args.repo / 'experiments/e27_preregistration.json').read_bytes()).hexdigest()
        paths = ['experiments/e27_execution.json', 'results/e27_report.md']
        for model in ('llama32_3b', 'llama31_8b'):
            root = args.repo / 'results' / model
            done = json.loads((root / 'E27_DONE.json').read_text())
            assert done['protocol_sha256'] == expected
            for name, digest in done['result_sha256'].items():
                assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
            paths.extend(str(p.relative_to(args.repo)) for p in root.glob('e27_*') if p.suffix in ('.csv', '.json') and '.partial.' not in p.name)
            paths.append(str((root / 'E27_DONE.json').relative_to(args.repo)))
        record.update(status='complete', artifact_hashes_verified=True)
        if args.push:
            assert run(['git', 'branch', '--show-current'], args.repo).stdout.strip() == 'main'
            old = run(['git', 'show', 'HEAD:report.md'], args.repo).stdout
            current = (args.repo / 'report.md').read_text()
            if without_e27(old) == without_e27(current):
                paths.append('report.md')
                record['main_report_included'] = True
            else:
                record['main_report_included'] = False
                record['main_report_note'] = 'Concurrent unrelated report edits left unstaged; E27 report also published separately in results/e27_report.md.'
            write_status(status, record)
            run(['git', 'add', '--', *paths], args.repo)
            run(['git', 'commit', '--only', '-m', 'Publish completed E27 paired subspace-selection results', '--', *paths], args.repo)
            run(['git', 'push', 'origin', 'main'], args.repo)
            commit = run(['git', 'rev-parse', 'HEAD'], args.repo).stdout.strip()
            remote = run(['git', 'ls-remote', 'origin', 'refs/heads/main'], args.repo).stdout.split()[0]
            assert remote == commit
            print('E27 complete and pushed:', commit, flush=True)
        else:
            write_status(status, record)


if __name__ == '__main__':
    main()
