#ifndef REF_RUNNER_H
#define REF_RUNNER_H
#include <fstream>

namespace ref {
// Correctness tests
void run_basic_operations_test();
void run_batch_test();
void run_random_operations_test();
void run_edge_case_test();

// Performance tests
void run_single_insertion_test(std::ofstream &csv_file);
void run_single_extraction_test(std::ofstream &csv_file);
void run_batch_insertion_test(std::ofstream &csv_file);
void run_batch_extraction_test(std::ofstream &csv_file);
} // namespace ref

#endif // REF_RUNNER_H
