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

void PriorityQueue::merge(std::vector<int>& arr, int left, int mid, int right) {
  int n1 = mid - left + 1;
  int n2 = right - mid;

  std::vector<int> L1(n1), L2(n2);
  // copy data to temp arrays L1 and L2
  for (int i = 0; i < n1; i++)L1[i] = arr[left + i];
  for (int i = 0; i < n2; i++)L2[i] = arr[right + i];

  int i = 0;
  int j = 0;
  int k = left;

  while(i < n1 && j < n2) {
    if(L1[i] <= L2[j]) {
      arr[k] = L1[i];
      i++;
    } else {
      arr[k] = L2[j];
      j++;
    }
    k++;
  }

  while(i < n1) {
    arr[k] = L1[i];
    i++;
    k++;
  }

  while(j < n2) {
    arr[k] = L2[j];
    j++;
    k++;
  }
  
}

void PriorityQueue::merge_sort(std::vector<int>& arr, int left, int right) {
  if(left <= right) return;
  int mid = left +(right - left) / 2 ;
  merge_sort(arr, left, mid);
  merge_sort(arr, mid + 1, right);
  merge(arr, left, mid, right);
}

void PriorityQueue::radix_sort(std::vector<int>& arr) {

}

void PriorityQueue::counting_sort(std::vector<int>& input, int exp) {
  int size = input.size();
  std::vector<int> output(size);
  int count[10] = {0};

  for(int i = 0; i < size; ++i) count[(input[i] / exp) % 10]++;
  for(int i = 1; i < 10; ++i) count[i] += count[i - 1];

  // Build the output array
  for (int i = size - 1; i >= 0; i--) {
      output[count[(input[i] / exp) % 10] - 1] = input[i];
      count[(input[i] / exp) % 10]--;
  } 


  for(int i = 0; i < size; ++i) input[i] = output[i]; // copy output array back
}

void PriorityQueue::radix_sort(std::vector<int>& arr) {
  int max = *std::max_element(arr.begin(), arr.end());
  for(int exp = 1; max / exp > 0; exp *= 10) counting_sort(arr, exp);
}

int PriorityQueue::size() const { return pq_size; }

bool PriorityQueue::isEmpty() const { return pq_size == 0; }

} // namespace ref_pq
