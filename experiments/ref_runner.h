#ifndef REF_RUNNER_H
#define REF_RUNNER_H

namespace ref {
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
} // namespace ref

#endif // REF_RUNNER_H
