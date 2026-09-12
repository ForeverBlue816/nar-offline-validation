from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as ce
for flag in ['-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_BFLOAT16_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__']:
    if flag in ce.COMMON_NVCC_FLAGS: ce.COMMON_NVCC_FLAGS.remove(flag)
setup(name='e28-v2-private-current-stream',ext_modules=[CUDAExtension(name='quarot._CUDA',sources=['quarot/kernels/'+f for f in ['bindings.cpp','gemm.cu','quant.cu','flashinfer.cu']],include_dirs=['/home/yanlongc/e28-v2-work/results/e28_v2/20260912_a40_v2_full_int4/local_build/quarot/kernels/include', '/projects/nar/nar-validation/external/quarot/third-party/cutlass/include', '/projects/nar/nar-validation/external/quarot/third-party/cutlass/tools/util/include'],extra_compile_args={'cxx':[],'nvcc':['-gencode','arch=compute_75,code=sm_75','-gencode','arch=compute_80,code=sm_80','-gencode','arch=compute_86,code=sm_86']})],cmdclass={'build_ext':BuildExtension})
