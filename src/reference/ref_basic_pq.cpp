#include "ref_basic_pq.h"
#include <algorithm>
#include <iostream>

namespace ref_pq {

PriorityQueue::PriorityQueue(int capacity) {
  this->capacity = capacity;
  pq_size = 0;
}

PriorityQueue::~PriorityQueue() {}

void PriorityQueue::insert(std::vector<PQNode> nodes) {
  if (pq_size + nodes.size() > capacity) {
    std::cout
        << "Priority Queue can't fit entire batch. No nodes were inserted."
        << std::endl;
    return;
  }
  for (int i = 0; i < nodes.size(); ++i) {
    pq.push_back(nodes[i]);
    pq_size++;
  }
  sort();
}

void PriorityQueue::insert(PQNode node) {
  if (pq_size + 1 > capacity) {
    std::cout << "Priority Queue is full. Node was not inserted." << std::endl;
    return;
  }
  pq.push_back(node);
  pq_size++;
  sort();
}

std::vector<PQNode> PriorityQueue::extract(int batch_size) {
  std::vector<PQNode> result;
  if (isEmpty()) {
    std::cout << "Priority Queue is empty" << std::endl;
    return result;
  }
  batch_size = std::min(batch_size, pq_size);
  for (int i = 0; i < batch_size; ++i) {
    result.push_back(pq[i]);
    pq_size--;
  }
  // NOTE: Maybe we should use a different data structure to store the priority,
  // however, this matches the GPU implementation.
  pq.erase(pq.begin(), pq.begin() + batch_size);
  return result;
}

PQNode PriorityQueue::extract() {
  PQNode result;
  if (isEmpty()) {
    std::cout << "Priority Queue is empty" << std::endl;
    return result;
  }
  result = pq[0];
  pq_size--;
  pq.erase(pq.begin());
  return result;
}

void PriorityQueue::sort() {
  std::sort(pq.begin(), pq.end(),
            [](const PQNode &a, const PQNode &b) { return a.key < b.key; });
}

int PriorityQueue::size() const { return pq_size; }

bool PriorityQueue::isEmpty() const { return pq_size == 0; }

} // namespace ref_pq
