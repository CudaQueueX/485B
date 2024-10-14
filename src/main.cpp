#include "gpu_priority_queue.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace gpu_pq;

// Helper function to generate random PQNodes
std::vector<PQNode> generateRandomNodes(int num_elements, int min_key = 0,
                                        int max_key = 1000) {
  std::vector<PQNode> nodes(num_elements);
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(min_key, max_key);

  for (int i = 0; i < num_elements; ++i) {
    nodes[i].key = distribution(generator);
    nodes[i].value = i; // Assigning index as value for identification
  }
  return nodes;
}

// Test 1: Basic Insertion and Extraction
void testBasicInsertionExtraction() {
  std::cout << "Running Test 1: Basic Insertion and Extraction...\n";

  int capacity = 1024;
  int num_elements = 512;

  PriorityQueue pq(capacity);

  auto nodes_to_insert = generateRandomNodes(num_elements);

  pq.insert(nodes_to_insert);

  auto extracted_nodes = pq.extract(num_elements);

  // Sort the original nodes for comparison
  std::sort(nodes_to_insert.begin(), nodes_to_insert.end(),
            [](const PQNode &a, const PQNode &b) { return a.key < b.key; });

  // Verify that the extracted nodes are in the correct order
  bool success = true;
  for (int i = 0; i < num_elements; ++i) {
    if (extracted_nodes[i].key != nodes_to_insert[i].key) {
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Test 1 Passed: Extracted nodes are in the correct order.\n";
  } else {
    std::cout
        << "Test 1 Failed: Extracted nodes are not in the correct order.\n";
  }
}

// Test 2: Insertion Beyond Capacity
void testInsertionBeyondCapacity() {
  std::cout << "Running Test 2: Insertion Beyond Capacity...\n";

  int capacity = 256;
  int num_elements = 300; // Exceeds capacity

  PriorityQueue pq(capacity);

  auto nodes_to_insert = generateRandomNodes(num_elements);

  // Attempt to insert nodes
  pq.insert(nodes_to_insert);

  int pq_size = pq.size();

  if (pq_size <= capacity) {
    std::cout << "Test 2 Passed: Priority queue did not exceed its capacity.\n";
  } else {
    std::cout << "Test 2 Failed: Priority queue exceeded its capacity.\n";
  }
}

// Test 3: Extraction from Empty Queue
void testExtractionFromEmptyQueue() {
  std::cout << "Running Test 3: Extraction from Empty Queue...\n";

  int capacity = 128;
  int extract_size = 50;

  PriorityQueue pq(capacity);

  // Attempt to extract nodes from an empty queue
  auto extracted_nodes = pq.extract(extract_size);

  if (extracted_nodes.empty()) {
    std::cout << "Test 3 Passed: No nodes extracted from an empty queue.\n";
  } else {
    std::cout << "Test 3 Failed: Nodes were extracted from an empty queue.\n";
  }
}

// Test 4: Concurrent Insertions
void testConcurrentInsertions() {
  std::cout << "Running Test 4: Concurrent Insertions...\n";

  int capacity = 1024;
  int num_elements = 800;

  PriorityQueue pq(capacity);

  auto nodes_to_insert = generateRandomNodes(num_elements);

  // Simulate concurrent insertions by dividing into batches
  int batch_size = 200;
  for (int i = 0; i < num_elements; i += batch_size) {
    int current_batch_size = std::min(batch_size, num_elements - i);
    std::vector<PQNode> batch(nodes_to_insert.begin() + i,
                              nodes_to_insert.begin() + i + current_batch_size);
    pq.insert(batch);
  }

  auto extracted_nodes = pq.extract(num_elements);

  // Sort the original nodes for comparison
  std::sort(nodes_to_insert.begin(), nodes_to_insert.end(),
            [](const PQNode &a, const PQNode &b) { return a.key < b.key; });

  // Verify that the extracted nodes are in the correct order
  bool success = true;
  for (int i = 0; i < extracted_nodes.size(); ++i) {
    if (extracted_nodes[i].key != nodes_to_insert[i].key) {
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Test 4 Passed: Concurrent insertions handled correctly.\n";
  } else {
    std::cout
        << "Test 4 Failed: Concurrent insertions not handled correctly.\n";
  }
}

// Test 5: Edge Case Values
void testEdgeCaseValues() {
  std::cout << "Running Test 5: Edge Case Values...\n";

  int capacity = 128;
  int num_elements = 128;

  PriorityQueue pq(capacity);

  // Create nodes with edge case priority values
  std::vector<PQNode> nodes_to_insert = {
      {INT_MIN, 1}, {INT_MAX, 2}, {0, 3}, {-1000, 4}, {1000, 5}};

  // Fill the rest with random nodes
  auto random_nodes =
      generateRandomNodes(num_elements - nodes_to_insert.size());
  nodes_to_insert.insert(nodes_to_insert.end(), random_nodes.begin(),
                         random_nodes.end());

  // Insert nodes
  pq.insert(nodes_to_insert);

  // Extract nodes
  auto extracted_nodes = pq.extract(num_elements);

  // Sort the original nodes for comparison
  std::sort(nodes_to_insert.begin(), nodes_to_insert.end(),
            [](const PQNode &a, const PQNode &b) { return a.key < b.key; });

  // Verify that the extracted nodes are in the correct order
  bool success = true;
  for (int i = 0; i < extracted_nodes.size(); ++i) {
    if (extracted_nodes[i].key != nodes_to_insert[i].key) {
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Test 5 Passed: Edge case values handled correctly.\n";
  } else {
    std::cout << "Test 5 Failed: Edge case values not handled correctly.\n";
  }
}

int main() {
  testBasicInsertionExtraction();
  std::cout << "----------------------------------------\n";
  testInsertionBeyondCapacity();
  std::cout << "----------------------------------------\n";
  testExtractionFromEmptyQueue();
  std::cout << "----------------------------------------\n";
  testConcurrentInsertions();
  std::cout << "----------------------------------------\n";
  testEdgeCaseValues();

  return 0;
}
