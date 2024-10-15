#include "cuda_basic_pq.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

namespace cuda_pq {

__global__ void insertKernel(PQNode *d_pq, PQNode *d_nodes, int batch_size,
                             int *d_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    int pos = atomicAdd(d_size, 1);
    d_pq[pos] = d_nodes[idx];
  }
}

__global__ void extractKernel(PQNode *d_pq, PQNode *d_nodes, int batch_size,
                              int *d_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    d_nodes[idx] = d_pq[idx];
    atomicSub(d_size, 1);
  }
}

__global__ void shiftLeftKernel(PQNode *d_input, PQNode *d_output,
                                int removed_size, int original_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= removed_size && idx < original_size) {
    d_output[idx - removed_size] = d_input[idx];
  }
}

__global__ void bitonicSortKernel(PQNode *d_pq, int size, int pass,
                                  int stride) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= size)
    return;

  unsigned int pairDistance = 1 << (stride - 1);
  unsigned int blockSize = 1 << pass;

  unsigned int leftIdx = idx;
  unsigned int rightIdx = idx ^ pairDistance;

  if (rightIdx > idx && rightIdx < size) {
    bool ascending = ((idx / blockSize) % 2) == 0;

    if ((d_pq[leftIdx].key > d_pq[rightIdx].key) == ascending) {
      // Swap elements
      PQNode temp = d_pq[leftIdx];
      d_pq[leftIdx] = d_pq[rightIdx];
      d_pq[rightIdx] = temp;
    }
  }
}

bool isPowerOf2(int n) { return n > 0 && (n & (n - 1)) == 0; }

int nextPowerOf2(int n) {
  int count = 0;
  if (n && !(n & (n - 1)))
    return n;

  while (n != 0) {
    n >>= 1;
    count += 1;
  }

  return 1 << count;
}

PriorityQueue::PriorityQueue(int capacity) {
  if (!isPowerOf2(capacity)) {
    capacity = nextPowerOf2(capacity);
  }
  this->capacity = capacity;
  d_size = 0;
  cudaMalloc(&d_size, sizeof(int));
  cudaMemset(d_size, 0, sizeof(int));
  h_size = new int;
  cudaMalloc(&d_pq, capacity * sizeof(PQNode));
  std::vector<PQNode> temp(capacity, PQ_MAX_VAL);
  cudaMemcpy(d_pq, temp.data(), capacity * sizeof(PQNode),
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_pq_buffer, capacity * sizeof(PQNode));
  cudaMemcpy(d_pq_buffer, temp.data(), capacity * sizeof(PQNode),
             cudaMemcpyHostToDevice);
}

PriorityQueue::~PriorityQueue() {
  cudaFree(d_pq);
  cudaFree(d_pq_buffer);
  cudaFree(d_size);
  delete h_size;
}

void PriorityQueue::insert(std::vector<PQNode> h_nodes) {
  if (size() + h_nodes.size() > capacity) {
    std::cout << "Priority queue can't fit batch. The nodes were not inserted."
              << std::endl;
    return;
  }

  int batch_size = h_nodes.size();
  PQNode *d_nodes;
  cudaMalloc(&d_nodes, batch_size * sizeof(PQNode));
  cudaMemcpy(d_nodes, h_nodes.data(), batch_size * sizeof(PQNode),
             cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
  cuda_pq::insertKernel<<<blocksPerGrid, threadsPerBlock>>>(d_pq, d_nodes,
                                                            batch_size, d_size);
  cudaDeviceSynchronize();
  cudaFree(d_nodes);
  sort();
}

void PriorityQueue::insert(PQNode h_node) {
  if (size() + 1 > capacity) {
    std::cout << "Priority queue is full. The node was not inserted."
              << std::endl;
    return;
  }
  PQNode *d_node;
  cudaMalloc(&d_node, sizeof(PQNode));
  cudaMemcpy(d_node, &h_node, sizeof(PQNode), cudaMemcpyHostToDevice);
  cuda_pq::insertKernel<<<1, 1>>>(d_pq, d_node, 1, d_size);
  cudaDeviceSynchronize();
  cudaFree(d_node);
  sort();
}

void PriorityQueue::swap_pq() {
  PQNode *temp = d_pq;
  d_pq = d_pq_buffer;
  d_pq_buffer = temp;
  std::vector<PQNode> temp_vec(capacity, PQ_MAX_VAL);
  cudaMemcpy(d_pq_buffer, temp_vec.data(), capacity * sizeof(PQNode),
             cudaMemcpyHostToDevice);
}

std::vector<PQNode> PriorityQueue::extract(int batch_size) {
  if (isEmpty()) {
    std::cout << "Priority queue is empty. No nodes were extracted."
              << std::endl;
    return std::vector<PQNode>();
  }
  int original_size = size();
  batch_size = std::min(batch_size, size());
  std::vector<PQNode> h_nodes(batch_size);
  PQNode *d_nodes;
  cudaMalloc(&d_nodes, batch_size * sizeof(PQNode));

  int threadsPerBlock = 256;
  int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
  cuda_pq::extractKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_pq, d_nodes, batch_size, d_size);
  cudaDeviceSynchronize();
  cudaMemcpy(h_nodes.data(), d_nodes, batch_size * sizeof(PQNode),
             cudaMemcpyDeviceToHost);
  cudaFree(d_nodes);
  int elements_to_shift = original_size - batch_size;
  blocksPerGrid = (elements_to_shift + threadsPerBlock - 1) / threadsPerBlock;
  cuda_pq::shiftLeftKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_pq, d_pq_buffer, batch_size, original_size);
  cudaDeviceSynchronize();
  swap_pq();
  return h_nodes;
}

PQNode PriorityQueue::extract() {
  if (isEmpty()) {
    std::cout << "Priority queue is empty. No node was extracted." << std::endl;
    return PQ_MAX_VAL;
  }
  int original_size = size();
  PQNode h_node;
  PQNode *d_node;
  cudaMalloc(&d_node, sizeof(PQNode));

  cuda_pq::extractKernel<<<1, 1>>>(d_pq, d_node, 1, d_size);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_node, d_node, sizeof(PQNode), cudaMemcpyDeviceToHost);
  cudaFree(d_node);
  int elements_to_shift = original_size - 1;
  int threadsPerBlock = 256;
  int blocksPerGrid =
      (elements_to_shift + threadsPerBlock - 1) / threadsPerBlock;
  cuda_pq::shiftLeftKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_pq, d_pq_buffer, 1, original_size);
  cudaDeviceSynchronize();
  swap_pq();
  return h_node;
}

void PriorityQueue::sort() {
  int size = this->size();
  if (!isPowerOf2(size)) {
    size = nextPowerOf2(size);
  }
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  int numPasses = __builtin_ctz(size);
  for (int pass = 1; pass <= numPasses; ++pass) {
    for (int stride = pass; stride > 0; --stride) {
      bitonicSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_pq, size, pass,
                                                            stride);
      cudaDeviceSynchronize();
    }
  }
}

int PriorityQueue::size() const {
  cudaMemcpy(h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);
  return *h_size;
}

bool PriorityQueue::isEmpty() const { return size() == 0; }

void PriorityQueue::print() const {
  int size = this->size();
  std::vector<PQNode> h_pq(size);
  std::cout << "Priority queue:\n";
  cudaMemcpy(h_pq.data(), d_pq, size * sizeof(PQNode), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    std::cout << "Node " << i << ": " << h_pq[i].key << " ";
    std::cout << h_pq[i].value << "\n";
  }
  std::cout << std::endl;
}
} // namespace cuda_pq
