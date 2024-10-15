#include "cuda_runner.h"
#include "ref_runner.h"
#include <iostream>

void run_correctness_tests() {
  std::cout << "Running correctness tests...\n\n";

  cuda::run_basic_operations_test();
  ref::run_basic_operations_test();

  cuda::run_batch_test();
  ref::run_batch_test();

  cuda::run_random_operations_test();
  ref::run_random_operations_test();

  cuda::run_edge_case_test();
  ref::run_edge_case_test();

  std::cout << "Correctness tests complete.\n\n";
}

int main() { run_correctness_tests(); }
