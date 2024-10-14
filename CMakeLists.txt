cmake_minimum_required(VERSION 3.20)

# Set compilers (ensure they are compatible with CUDA)
set(CMAKE_C_COMPILER /usr/bin/gcc-13)
set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler")

# Project declaration
project(CudaTest LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Source files
set(SOURCES
    src/gpu_priority_queue.cu
    src/main.cpp
)

# Add executable
add_executable(CudaPQTest ${SOURCES})

# Include CUDA directories
target_include_directories(CudaPQTest PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(CudaPQTest PRIVATE ${CUDAToolkit_LIBRARIES})

# Target properties
set_target_properties(CudaPQTest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80"
)

