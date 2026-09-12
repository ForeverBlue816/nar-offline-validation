"""Run the one bounded graph adapter only after the original pipeline finishes."""
from .common import *
import time


def main():
    p = arguments(__doc__)
    p.add_argument('--wait-for-primary', action='store_true')
    a = p.parse_args()
    out = a.run
    if a.wait_for_primary:
        while not (out / 'pipeline_finished.json').exists():
            time.sleep(20)
    def run(module, *args):
        cmd = [sys.executable, '-m', 'e28_v2.' + module, *map(str, args), '--run', str(out)]
        with (out / 'stream_commands.jsonl').open('a') as f:
            f.write(json.dumps({'timestamp': now(), 'argv': cmd}) + '\n')
        return subprocess.run(cmd, check=False).returncode
    manifest = out / 'stream_backend_manifest.json'
    if not manifest.exists() or read(manifest).get('status') != 'BUILT':
        write(out / 'stream_pipeline_finished.json', {'status': 'BLOCKED', 'reason': 'The single isolated stream-adapter build is unavailable', 'timestamp': now()})
        return
    for backend in ['original', 'current_stream']:
        run('backend_fingerprint', '--backend', backend)
    original = read(out / 'correctness/backend_fingerprint_original.json')
    patched = read(out / 'correctness/backend_fingerprint_current_stream.json')
    exact = original.get('status') == patched.get('status') == 'COMPLETE' and bool(original.get('extension_calls')) and original['extension_calls'] == patched['extension_calls']
    audit = {'timestamp': now(), 'status': 'PASS' if exact else 'FAIL', 'extension_output_storage_exact': exact,
             'original': 'correctness/backend_fingerprint_original.json', 'patched': 'correctness/backend_fingerprint_current_stream.json',
             'interpretation': 'Equivalence of actual outputs/storage on the original operator suite; known numerical boundary failures remain failures in both backends.'}
    write(out / 'stream_backend_equivalence.json', audit)
    if not exact:
        write(out / 'stream_pipeline_finished.json', {'status': 'BLOCKED', 'reason': 'Current-stream outputs/storage did not match original extension; no graph performance ranking', 'timestamp': now()})
        return
    eligible = []
    for model in MODELS:
        for method in METHODS:
            run('graph', '--model', model, '--method', method, '--backend', 'current_stream')
        if all((out / 'correctness' / f'graph_stream_{model}_{method}.json').exists() and read(out / 'correctness' / f'graph_stream_{model}_{method}.json').get('status') == 'PASS' for method in METHODS):
            eligible.append(model)
    for session, order in enumerate(read(out / 'protocol.json')['method_order'], 1):
        for model in eligible:
            for method in order:
                for mode in (['eager_sequence','cuda_graph_sequence'] if session != 2 else ['cuda_graph_sequence','eager_sequence']):
                    run('graph_benchmark', '--model', model, '--method', method, '--mode', mode, '--session', session)
        run('collect')
    write(out / 'stream_pipeline_finished.json', {'status': 'FINISHED', 'eligible_models': eligible, 'timestamp': now(), 'note': 'See individual statuses; finite/metadata failures are not hidden by this completion marker.'})
    run('collect')


if __name__ == '__main__': main()
