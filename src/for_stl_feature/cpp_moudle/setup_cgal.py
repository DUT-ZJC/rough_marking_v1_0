# src/for_stl_feature/cpp_module/setup_cgal.py
import os
import sys
from setuptools import setup, Extension
import pybind11

conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)

# 自动定位 Anaconda 环境下的头文件和库路径
include_dirs = [
    pybind11.get_include(),
    os.path.join(conda_prefix, 'Library', 'include')
]
library_dirs = [
    os.path.join(conda_prefix, 'Library', 'lib')
]

if sys.platform == 'win32':
    extra_compile_args = [
        '/O2', 
        '/EHsc', 
        '/std:c++17', 
        '/Zc:__cplusplus',
        '/utf-8',
        '/permissive-',  
    ]
    # Windows 下库名字没有 lib 前缀
    cgal_libs = ["gmp", "mpfr"]
else:
    extra_compile_args = ['-O3', '-std=c++17']
    cgal_libs = ["gmp", "mpfr"]

ext_modules = [
    Extension(
        "cgal_ransac",
        ["cgal_ransac.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=[
            ('CGAL_HEADER_ONLY', '1'), 
            ('BOOST_ALL_NO_LIB', '1'),  
            ('_USE_MATH_DEFINES', None),
            ('NOMINMAX', None),         
        ],
        libraries=cgal_libs,  # ✨ 核心修复点：去掉了 lib 前缀
        extra_compile_args=extra_compile_args
    ),
]

setup(
    name="cgal_ransac",
    ext_modules=ext_modules,
)