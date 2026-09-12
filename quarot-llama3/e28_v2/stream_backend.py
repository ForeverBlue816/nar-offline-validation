"""One bounded current-stream adapter, built privately without installing packages."""
from .common import *
import difflib, shutil


def activate(run):
    manifest = read(run / 'stream_backend_manifest.json')
    if manifest.get('status') != 'BUILT':
        raise RuntimeError('The isolated current-stream backend has not built successfully')
    local = run / 'local_build'
    binary = pathlib.Path(manifest['binary'])
    if sha(binary) != manifest['binary_sha256']:
        raise RuntimeError('Current-stream binary changed after the build manifest')
    if any(name == 'quarot' or name.startswith('quarot.') for name in sys.modules):
        raise RuntimeError('Select backend before importing QuaRot')
    sys.path.insert(0, str(local))
    return manifest


def main():
    a = arguments(__doc__).parse_args()
    out = a.run.resolve()
    dest = out / 'stream_backend_manifest.json'
    if dest.exists():
        print('EXISTS', dest, flush=True)
        return
    protocol_file=out/'protocol_stream_addendum.json'
    if not protocol_file.exists():
        write(protocol_file,{'frozen_at':now(),'stage':'P1 conditional bounded adapter',
              'attempts':'One current-stream patch to existing quantization/KV/CUTLASS dispatch; no arithmetic/tile/dtype change or tuning.',
              'validation':'Exact original/patched extension output and KV storage fingerprints, followed by all-three-method A-B-A growing-cache verification with exact cache and hidden/logit relativeL2<=0.002.',
              'timing_condition':'Only after every method of a model passes: same patched binary eager and graph,10 complete warmups+50 runs+3 balanced sessions,2048prefix+128steps discard8; metadata updates included.',
              'method_order':read(out/'protocol.json')['method_order'],
              'window_start_offsets':[0,4096],
              'scope':'Separate raw_graph_runs; never pooled with original-binary core measurements. Known accumulator narrowing and native-paper-format limits remain unchanged.'})
    source = WORK / 'external/quarot'
    local = out / 'local_build'
    local.mkdir(exist_ok=True)
    (local / '.gitignore').write_text('*\n!.gitignore\n')
    shutil.copytree(source / 'quarot', local / 'quarot',
                    ignore=shutil.ignore_patterns('*.so', '__pycache__'), dirs_exist_ok=False)
    changes = []
    expected = {'quant.cu': 2, 'flashinfer.cu': 6, 'gemm.cu': 1}
    for name, count in expected.items():
        target = local / 'quarot/kernels' / name
        before = target.read_text()
        if name == 'gemm.cu':
            old, new = 'gemmOp(arguments)', 'gemmOp(arguments, nullptr, at::cuda::getCurrentCUDAStream())'
        else:
            old = '<<<grid, block>>>' if name == 'quant.cu' else '<<<nblks, nthrs>>>'
            new = old[:-3] + ', 0, at::cuda::getCurrentCUDAStream()>>>'
        if before.count(old) != count:
            raise RuntimeError(f'Unexpected upstream launch count in {name}')
        after = '#include <ATen/cuda/CUDAContext.h>\n' + before.replace(old, new)
        target.write_text(after)
        changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                       fromfile='a/quarot/kernels/' + name, tofile='b/quarot/kernels/' + name))
    patch = out / 'current_stream_backend.patch'
    patch.write_text(''.join(changes))
    setup = '''from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as ce
for flag in ['-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_BFLOAT16_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__']:
    if flag in ce.COMMON_NVCC_FLAGS: ce.COMMON_NVCC_FLAGS.remove(flag)
setup(name='e28-v2-private-current-stream',ext_modules=[CUDAExtension(name='quarot._CUDA',sources=['quarot/kernels/'+f for f in ['bindings.cpp','gemm.cu','quant.cu','flashinfer.cu']],include_dirs=INCLUDES,extra_compile_args={'cxx':[],'nvcc':['-gencode','arch=compute_75,code=sm_75','-gencode','arch=compute_80,code=sm_80','-gencode','arch=compute_86,code=sm_86']})],cmdclass={'build_ext':BuildExtension})
'''
    includes = [str(local / 'quarot/kernels/include'), str(source / 'third-party/cutlass/include'),
                str(source / 'third-party/cutlass/tools/util/include')]
    setup = setup.replace('INCLUDES', repr(includes))
    (local / 'setup_stream.py').write_text(setup)
    (out / 'setup_stream_exact.py').write_text(setup)
    manifest = {'started': now(), 'status': 'BUILDING', 'source_quarot_commit': command(['git','-C',str(source),'rev-parse','HEAD']),
                'source_extension': [{'path': str(p), 'sha256': sha(p)} for p in (source / 'quarot').glob('_CUDA*.so')],
                'patch_sha256': sha(patch), 'scope': 'Only launch stream: two quantization launches, six existing KV launches, and existing CUTLASS stream argument. No arithmetic, tile, dtype, quantizer or attention changes.',
                'nvcc': command(['nvcc','--version']), 'compiler': command(['g++','--version']),
                'setup_sha256': sha(out / 'setup_stream_exact.py'), 'host': command(['hostname']),
                'command': [sys.executable, 'setup_stream.py', 'build_ext', '--inplace'],
                'environment': {k:os.environ.get(k) for k in ('CUDA_HOME','MAX_JOBS','SLURM_JOB_ID','SLURM_CPUS_PER_TASK')}}
    write(dest, manifest)
    with (out / 'stream_backend_build.log').open('w') as log:
        rc = subprocess.run(manifest['command'], cwd=local, stdout=log, stderr=subprocess.STDOUT).returncode
    binaries = list((local / 'quarot').glob('_CUDA*.so'))
    manifest.update(ended=now(), returncode=rc, status='BUILT' if rc == 0 and len(binaries) == 1 else 'BLOCKED')
    if manifest['status'] == 'BUILT':
        manifest.update(binary=str(binaries[0]), binary_sha256=sha(binaries[0]))
    else:
        manifest['reason'] = 'Bounded isolated build failed; see stream_backend_build.log'
    write(dest, manifest)
    print(manifest['status'], flush=True)


if __name__ == '__main__':
    main()
