# Collect CUDA and reference source files
file(GLOB CUDA_SOURCES "cuda/*.cu" "cuda/*.cpp" "cuda/*.h")
file(GLOB REFERENCE_SOURCES "reference/*.cpp" "reference/*.h")

# Create a library from the source files
add_library(src_library ${CUDA_SOURCES} ${REFERENCE_SOURCES})

# Include CUDA and reference directories
target_include_directories(src_library PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda 
    ${CMAKE_CURRENT_SOURCE_DIR}/reference 
    ${CUDAToolkit_INCLUDE_DIRS}  # Include CUDA headers
)

# Link CUDA libraries
target_link_libraries(src_library PRIVATE ${CUDAToolkit_LIBRARIES})

