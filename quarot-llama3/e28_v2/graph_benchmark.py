"""Independent-process matched eager/graph timings for the private stream adapter."""
from .common import *
import gc, time, types


def main():
    p = arguments(__doc__)
    p.add_argument('--mode', choices=['eager_sequence', 'cuda_graph_sequence'], required=True)
    p.add_argument('--session', type=int, required=True)
    a = p.parse_args()
    key = f'{a.model}_{a.method}_{a.mode}_decode_s{a.session}'
    dest = a.run / 'raw_graph_runs' / (key + '.json')
    if dest.exists():
        print('EXISTS', dest, flush=True)
        return
    checks = [a.run / 'correctness' / f'graph_stream_{a.model}_{method}.json' for method in METHODS]
    if any(not path.exists() or read(path).get('status') != 'PASS' for path in checks):
        write(dest, {'key': key, 'status': 'BLOCKED', 'reason': 'All three methods must pass growing-cache verification before a comparable graph panel'})
        return
    from .stream_backend import activate
    backend = activate(a.run)
    import torch
    from .cache_adapter import make_cache, clear, snapshot
    from .benchmark import finite_check
    initialize()
    result = {'key': key, 'model': a.model, 'method': a.method, 'mode': a.mode, 'phase': 'decode',
              'session': a.session, 'started': now(), 'status': 'RUNNING', 'environment': environment(),
              'backend_manifest': backend, 'protocol_stream_sha256': sha(a.run / 'protocol_stream_addendum.json'),
              'scope': 'Private current-stream backend, matched eager/graph panel only; never pooled with original binary core records'}
    write(dest, result)
    try:
        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            model = build(a.model, a.method)
            for layer in model.model.layers:
                def full_cache(self, x, seq_len=None):
                    return self.cos_cached, self.sin_cached
                layer.self_attn.rotary_emb.forward = types.MethodType(full_cache, layer.self_attn.rotary_emb)
            result['loaded'] = mempoint()
            result['shared_state'] = state_hashes(model)
            generator = torch.Generator(device='cpu').manual_seed(0)
            ids = torch.randint(100, 200, (1, 2048), generator=generator, dtype=torch.int32).cuda()
            token = torch.full((1, 1), 100, device='cuda', dtype=torch.int32)
            cache = make_cache(model, 1, 2176)
            result.update(batch=1, prefix=2048, capacity=2176, input_sha256=tensor_hash(ids), decode_token=100)
            result['finite'] = finite_check(model, ids, cache, tail_token=token)
            if result['finite']['status'] != 'PASS':
                result.update(status='INVALID', reason='Nonfinite preflight')
                return
            def prefix():
                cache.graph_mode = False
                clear(cache)
                model(ids, past_key_values=cache)
            # Warm both TorchScript packing shapes before exact cache comparisons.
            for _ in range(2):
                prefix()
                for _ in range(2): model(token, past_key_values=cache)
            prefix()
            expected_prefix = snapshot(cache)
            prefix()
            if snapshot(cache) != expected_prefix:
                raise RuntimeError('Prefix reset is not exact')
            graph = None
            static = {}
            if a.mode == 'cuda_graph_sequence':
                position = torch.empty((1, 1), device='cuda', dtype=torch.int64)
                spec = {'kv_data': cache.pages, 'kv_param': cache.scales,
                        'kv_indices': torch.zeros(1, device='cuda', dtype=torch.int32),
                        'kv_indptr': torch.tensor([0, 1], device='cuda', dtype=torch.int32),
                        'last_page_offset': torch.empty(1, device='cuda', dtype=torch.int32)}
                static = {'position': position, **{k: v for k, v in spec.items() if k not in ('kv_data', 'kv_param')}}
                def metadata(length):
                    spec['kv_indptr'][1:].fill_(1)
                    spec['last_page_offset'].fill_(length)
                    position.fill_(length - 1)
                def graph_prefix():
                    prefix()
                    cache._spec = spec
                    cache.graph_mode = True
                    metadata(2049)
                graph_prefix()
                preparation_start = time.perf_counter()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3): model(token, past_key_values=cache, position_ids=position)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph_prefix()
                before = {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = model(token, past_key_values=cache, position_ids=position)
                torch.cuda.synchronize()
                after = {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
                result['graph_preparation'] = {'seconds': time.perf_counter() - preparation_start, 'graphs': 1,
                        'allocated_delta_bytes': after['allocated'] - before['allocated'],
                        'reserved_delta_bytes': after['reserved'] - before['reserved'],
                        'pool_identifier': list(graph.pool()),
                        'note': 'Private graph-pool deltas are separate from total independent-process inference peaks.'}
                # This timing harness also checks its own replay, using the exact
                # measured random input and all page/scales bytes at selected steps.
                prefix()
                reference = {}
                for step in range(1, 129):
                    observed = model(token, past_key_values=cache)
                    if step in (1, 2, 9, 64, 65, 128):
                        reference[step] = (observed.logits.clone(), snapshot(cache))
                graph_prefix()
                replay_checks = []
                for step in range(1, 129):
                    metadata(2048 + step)
                    graph.replay()
                    cache.length = 2048 + step
                    if step in reference:
                        expected, expected_cache = reference[step]
                        finite = bool(torch.isfinite(output.logits).all())
                        rel = float((output.logits.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)) if finite else None
                        observed_cache = snapshot(cache)
                        exact = observed_cache == expected_cache
                        replay_checks.append({'step': step, 'finite': finite, 'relative_l2': rel, 'cache_exact': exact, 'expected_cache': expected_cache, 'observed_cache': observed_cache,
                                              'status': 'PASS' if finite and rel <= .002 and exact else 'FAIL'})
                result['timing_harness_replay_checks'] = replay_checks
                if any(r['status'] != 'PASS' for r in replay_checks):
                    result.update(status='INVALID', reason='Timing harness replay validation failed')
                    return
                del reference, expected, observed
            else:
                result['graph_preparation'] = {'seconds': None, 'graphs': 0, 'reason': 'Eager process does not capture or retain a graph pool'}

            def run(events=False):
                if graph is None:
                    prefix()
                    for _ in range(8): model(token, past_key_values=cache)
                else:
                    graph_prefix()
                    for step in range(1, 9):
                        metadata(2048 + step)
                        graph.replay()
                        cache.length = 2048 + step
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True) if events else None
                end_event = torch.cuda.Event(enable_timing=True) if events else None
                if events: start_event.record()
                start = time.perf_counter()
                if graph is None:
                    for _ in range(120): model(token, past_key_values=cache)
                else:
                    for step in range(9, 129):
                        metadata(2048 + step)
                        graph.replay()
                        cache.length = 2048 + step
                if events: end_event.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                return elapsed, start_event.elapsed_time(end_event) if events else None
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            for _ in range(10): run()
            result['warmed'] = mempoint()
            result['storage'] = storage_breakdown(model, cache, {'input_ids': ids, 'token': token, **static})
            torch.cuda.reset_peak_memory_stats()
            result['samples'] = []
            for i in range(50):
                stamp = now()
                elapsed, _ = run()
                result['samples'].append({'run': i, 'timestamp': stamp, 'elapsed_s': elapsed, 'ms_per_token': elapsed * 1000 / 120})
                if i % 10 == 0: print('GRAPH_PANEL', key, i, round(elapsed, 5), flush=True)
            result['inference_peak'] = mempoint()
            _, result['cuda_event_elapsed_ms'] = run(events=True)
            result['summary'] = stats([r['ms_per_token'] for r in result['samples']])
            result.update(status='PASS', environment_after=environment())
    except Exception as exc:
        import traceback
        result.update(status='FAIL', reason=repr(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        result['ended'] = now()
        write(dest, result)


if __name__ == '__main__':
    main()
