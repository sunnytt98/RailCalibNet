import glob
import os.path as osp
import platform
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = (
    sorted(glob.glob(osp.join(ROOT_DIR, "*.cpp"))) +
    sorted(glob.glob(osp.join(ROOT_DIR, "*.cu")))
)

if platform.system() == "Windows":
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17"],
        "nvcc": ["-O2", "-std=c++17"],
    }
else:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
    }


setup(
    name='lidar_pro',
    version='0.1.0',
    author='kwea123',
    author_email='kwea123@gmail.com',
    description='LiDAR projection CUDA ops',
    long_description='LiDAR projection CUDA ops for depth/image conversion',
    packages=find_packages(),
    python_requires='>=3.8',
    ext_modules=[
        CUDAExtension(
            name='lidar_pro._C',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
