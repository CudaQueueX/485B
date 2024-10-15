cmake_minimum_required(VERSION 3.20)

# Set compilers
set(CMAKE_C_COMPILER /usr/bin/gcc-13)
set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project declaration
project(GPUComputingProject LANGUAGES CXX CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Add src and experiments directories
add_subdirectory(src)
add_subdirectory(experiments)

