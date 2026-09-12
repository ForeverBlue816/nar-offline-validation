"""Hash actual extension outputs/mutated KV storage in the Level-2 operator suite."""
from .common import *


def main():
    p = arguments(__doc__)
    p.add_argument('--backend', choices=['original', 'current_stream'], required=True)
    a = p.parse_args()
    dest = a.run / 'correctness' / f'backend_fingerprint_{a.backend}.json'
    if dest.exists(): return
    if a.backend == 'current_stream':
        from .stream_backend import activate
        activate(a.run)
    import torch, quarot
    from .verify import operators
    initialize()
    records = []
    names = ['matmul', 'sym_quant', 'sym_dequant', 'init_kv_i4', 'append_kv_i4', 'batch_decode_i4',
             'init_kv_f16', 'append_kv_f16', 'batch_decode_f16']
    def wrap(name, original):
        def call(*args, **kwargs):
            value = original(*args, **kwargs)
            tensors = {'return': value} if torch.is_tensor(value) else {}
            if name.startswith(('init_kv_', 'append_kv_')):
                tensors.update(kv_data=args[0], kv_params=args[1])
            elif name.startswith('batch_decode_'):
                tensors['output'] = args[0]
            records.append({'operator': name, 'outputs': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': tensor_hash(v)} for k, v in tensors.items()}})
            return value
        return call
    result = {'started': now(), 'environment': environment(), 'backend': a.backend,
              'binary': quarot._CUDA.__file__, 'binary_sha256': sha(quarot._CUDA.__file__)}
    try:
        for name in names:
            setattr(quarot._CUDA, name, wrap(name, getattr(quarot._CUDA, name)))
        with torch.inference_mode():
            suite = operators(a)
        result.update(status='COMPLETE', operator_suite=suite, extension_calls=records,
                      note='COMPLETE means fingerprints collected; the preserved suite includes the known accumulator-narrowing failures.')
    except Exception as exc:
        import traceback
        result.update(status='BLOCKED', reason=repr(exc), traceback=traceback.format_exc())
    result['ended'] = now()
    write(dest, result)


if __name__ == '__main__': main()
