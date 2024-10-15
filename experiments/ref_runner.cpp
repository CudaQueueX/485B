#include "ref_runner.h"
#include "../src/reference/ref_basic_pq.h"
#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>

namespace ref {

/****************************************************
                  CORRECTNESS TESTS
*****************************************************/

void run_basic_operations_test() {
  int capacity = 10;
  ref_pq::PriorityQueue pq(capacity);

  // Insertion
  PQNode node1{5, 1};
  PQNode node2{3, 2};

  pq.insert(node1);
  pq.insert(node2);

  // Extraction
  PQNode extracted_node1 = pq.extract();
  PQNode extracted_node2 = pq.extract();

  // Verification
  std::cout << "Running REFERENCE Test ...\n";
  if (extracted_node1.key == 3 && extracted_node1.value == 2 &&
      extracted_node2.key == 5 && extracted_node2.value == 1) {
    std::cout << "Test 1 Passed: ";
  } else {
    std::cout << "Test 1 Failed: ";
  }
  std::cout << "Basic Insertion and Extraction.\n" << std::endl;
}

void run_batch_test() {
  int capacity = 100;
  ref_pq::PriorityQueue pq(capacity);

  // Batch Insertion
  std::vector<PQNode> nodes;
  for (int i = 0; i < capacity; ++i) {
    nodes.push_back(PQNode{std::rand(), i});
  }
  pq.insert(nodes);

  // Extraction
  std::vector<PQNode> extracted_nodes = pq.extract(capacity);

  // Verification
  // Sort the original nodes for comparison
  std::sort(nodes.begin(), nodes.end(),
            [](const PQNode &a, const PQNode &b) { return a.key < b.key; });

  bool success = true;
  for (int i = 0; i < capacity; ++i) {
    if (extracted_nodes[i].key != nodes[i].key) {
      success = false;
      break;
    }
  }

  std::cout << "Running REFERENCE Test ...\n";
  if (success) {
    std::cout << "Test 2 Passed: ";
  } else {
    std::cout << "Test 2 Failed: ";
  }
  std::cout << "Batch Insertion and Extraction.\n" << std::endl;
}

void run_random_operations_test() {
  int capacity = 100;
  ref_pq::PriorityQueue pq(capacity);

  // Insertion of random number of elements
  int num_to_insert = 42;
  std::vector<PQNode> nodes;
  for (int i = 0; i < num_to_insert; ++i) {
    nodes.push_back(PQNode{std::rand(), i});
  }
  pq.insert(nodes);

  // Extraction of random number of elements
  int num_to_extract = 23;
  std::vector<PQNode> extracted_nodes_batch1 = pq.extract(num_to_extract);

  // Verification
  // Sort the original nodes for comparison
  std::sort(nodes.begin(), nodes.end(),
            [](const PQNode &a, const PQNode &b) { return a.key < b.key; });

  bool success = true;
  for (int i = 0; i < num_to_extract; ++i) {
    if (extracted_nodes_batch1[i].key != nodes[i].key) {
      success = false;
      break;
    }
  }

  // Extraction of remaining elements
  std::vector<PQNode> extracted_nodes_batch2 =
      pq.extract(num_to_insert - num_to_extract);

  // Verification
  for (int i = 0; i < num_to_insert - num_to_extract; ++i) {
    if (extracted_nodes_batch2[i].key != nodes[i + num_to_extract].key) {
      success = false;
      break;
    }
  }

  // Verify that the priority queue is empty
  if (!pq.isEmpty()) {
    success = false;
  }
  std::cout << "Running REFERENCE Test ...\n";
  if (success) {
    std::cout << "Test 3 Passed: ";
  } else {
    std::cout << "Test 3 Failed: ";
  }
  std::cout << "Random Insertion and Extraction.\n" << std::endl;
}

void run_edge_case_test() {
  int capacity = 100;
  ref_pq::PriorityQueue pq(capacity);

  // Insertion of elements with the same key
  int num_to_insert = 10;
  int shared_key = 42;
  std::vector<PQNode> nodes;
  for (int i = 0; i < num_to_insert; ++i) {
    nodes.push_back(PQNode{shared_key, i});
  }
  pq.insert(nodes);

  // Extraction
  std::vector<PQNode> extracted_nodes = pq.extract(num_to_insert);

  // Verification
  bool success = true;
  for (int i = 0; i < num_to_insert; ++i) {
    if (extracted_nodes[i].key != shared_key) {
      success = false;
      break;
    }
  }

  std::cout << "Running REFERENCE Test ...\n";
  if (success) {
    std::cout << "Test 4 Passed: ";
  } else {
    std::cout << "Test 4 Failed: ";
  }
  std::cout << "Edge Case: Insertion of Elements with the Same Key.\n"
            << std::endl;

  // Insertion of minimum and maximum priority values
  PQNode min_node{INT_MIN, 0};
  PQNode max_node{INT_MAX, 1};
  PQNode zero_node{0, 2};
  PQNode neg_thousand_node{-1000, 3};

  pq.insert(min_node);
  pq.insert(max_node);
  pq.insert(zero_node);
  pq.insert(neg_thousand_node);

  PQNode extracted_min_node = pq.extract();
  PQNode extract_neg_thousand_node = pq.extract();
  PQNode extracted_zero_node = pq.extract();
  PQNode extracted_max_node = pq.extract();

  // Verification
  success = true;
  if (extracted_min_node.key != INT_MIN || extracted_min_node.value != 0) {
    std::cout << "Failed Min Key" << std::endl;
    std::cout << "extracted_min_node.key: " << extracted_min_node.key
              << std::endl;
    std::cout << "extracted_min_node.value: " << extracted_min_node.value
              << "\n"
              << std::endl;
    success = false;
  }
  if (extracted_max_node.key != INT_MAX || extracted_max_node.value != 1) {
    std::cout << "Failed Max Key" << std::endl;
    std::cout << "extracted_max_node.key: " << extracted_max_node.key
              << std::endl;
    std::cout << "extracted_max_node.value: " << extracted_max_node.value
              << "\n"
              << std::endl;
    success = false;
  }
  if (extracted_zero_node.key != 0 || extracted_zero_node.value != 2) {
    std::cout << "Failed Zero Key" << std::endl;
    std::cout << "extracted_zero_node.key: " << extracted_zero_node.key
              << std::endl;
    std::cout << "extracted_zero_node.value: " << extracted_zero_node.value
              << "\n"
              << std::endl;
    success = false;
  }
  if (extract_neg_thousand_node.key != -1000 ||
      extract_neg_thousand_node.value != 3) {
    std::cout << "Failed Neg Thousand Key" << std::endl;
    std::cout << "extract_neg_thousand_node.key: "
              << extract_neg_thousand_node.key << std::endl;
    std::cout << "extract_neg_thousand_node.value: "
              << extract_neg_thousand_node.value << "\n"
              << std::endl;
    success = false;
  }

  std::cout << "Running REFERENCE Test ...\n";
  if (success) {
    std::cout << "Test 5 Passed: ";
  } else {
    std::cout << "Test 5 Failed: ";
  }
  std::cout << "Edge Case: Minimum and Maximum Priority Values.\n" << std::endl;
}

/****************************************************
                  PERFORMANCE TESTS
*****************************************************/
} // namespace ref
