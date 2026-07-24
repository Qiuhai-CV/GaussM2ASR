"""Build the GaussM2ASR CUDA Gaussian-splatting extensions in place."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent
if CUDA_HOME is None:
    raise RuntimeError(
        'CUDA Toolkit was not found. Install a toolkit with nvcc, add it to PATH, '
        'and run this command again.')



def source_paths(renderer_directory):
    renderer_root = ROOT / 'basicsr' / 'utils' / renderer_directory
    return [str(renderer_root / 'gswrapper.cpp'), str(renderer_root / 'gs.cu')]


setup(
    name='gaussm2asr-gscuda',
    ext_modules=[
        CUDAExtension(
            name='basicsr.utils.gs_cuda.gscuda',
            sources=source_paths('gs_cuda'),
        ),
        CUDAExtension(
            name='basicsr.utils.gs_cuda_dmax.gscuda',
            sources=source_paths('gs_cuda_dmax'),
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
