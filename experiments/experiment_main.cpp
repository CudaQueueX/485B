#include "cuda_runner.h"
#include "ref_runner.h"
#include <fstream>
#include <iostream>
#include <string>

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

std::ofstream open_csv_file(const std::string &filename) {
  std::ofstream file(filename, std::ios::out | std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open CSV file for writing." << std::endl;
  }
  return file;
}

void write_header(std::ofstream &file) {
  file.seekp(0, std::ios::end);
  if (file.tellp() != 0) {
    return;
  }
  file << "test_type,num_operations,run,time_ms,batch_size\n";
}

void run_performance_tests() {
  std::ofstream cuda_csv_file = open_csv_file("cuda_performance_results.csv");
  write_header(cuda_csv_file);

  std::ofstream ref_csv_file = open_csv_file("ref_performance_results.csv");
  write_header(ref_csv_file);

  std::cout << "Running performance tests...\n\n";
  cuda::run_single_insertion_test(cuda_csv_file);
  ref::run_single_insertion_test(ref_csv_file);

  cuda::run_single_extraction_test(cuda_csv_file);
  ref::run_single_extraction_test(ref_csv_file);

  cuda::run_batch_insertion_test(cuda_csv_file);
  ref::run_batch_insertion_test(ref_csv_file);

  cuda::run_batch_extraction_test(cuda_csv_file);
  ref::run_batch_extraction_test(ref_csv_file);
  std::cout << "Performance tests complete.\n\n";

  cuda_csv_file.close();
  ref_csv_file.close();
}

int main() {
  run_correctness_tests();
  run_performance_tests();
}
