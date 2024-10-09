cmake_minimum_required(VERSION 3.16)

# Explicitly specify the C and C++ compilers before project()
set(CMAKE_C_COMPILER /usr/bin/gcc-13)
set(CMAKE_CXX_COMPILER /usr/bin/g++-13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Specify the CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)

# Set CUDA flags before project()
set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler")

project(CudaTest LANGUAGES CXX CUDA)

# Export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SOURCES
    src/*.cu
)

add_executable(CudaTest ${SOURCES})

target_link_libraries(CudaTest PRIVATE ${CMAKE_CUDA_LIBRARIES})

set_target_properties(CudaTest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80"
)

