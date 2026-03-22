cmake_minimum_required(VERSION 3.12)
project(FastRansac)

# 寻找 pybind11 (可通过 pip install pybind11 获得)
find_package(pybind11 REQUIRED)

# 编译为 Python 扩展模块
pybind11_add_module(fast_ransac_cpp ransac_wrapper.cpp /*你的其他C++源文件*/)