#ifndef CUDA_RUNNER_H
#define CUDA_RUNNER_H

namespace cuda {
// Correctness tests
void run_basic_operations_test();
void run_batch_test();
void run_random_operations_test();
void run_edge_case_test();

// Performance tests
void run_large_throughput_test();
void run_medium_throughput_test();
void run_small_throughput_test();
void run_single_insertion_test();
void run_single_extraction_test();
} // namespace cuda

#endif // CUDA_RUNNER_H
