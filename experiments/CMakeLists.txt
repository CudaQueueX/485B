# Collect experiment source files
set(EXPERIMENT_SOURCES 
    experiment_main.cpp 
    cuda_runner.cpp 
    ref_runner.cpp
)

# Create the ExperimentHarness executable
add_executable(ExperimentHarness ${EXPERIMENT_SOURCES})

# Link src_library with ExperimentHarness
target_link_libraries(ExperimentHarness PRIVATE src_library)

