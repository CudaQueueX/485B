#include "gpu_priority_queue.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

namespace gpu_pq {

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
}

PriorityQueue::~PriorityQueue() {
  cudaFree(d_pq);
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
  gpu_pq::insertKernel<<<blocksPerGrid, threadsPerBlock>>>(d_pq, d_nodes,
                                                           batch_size, d_size);
  cudaDeviceSynchronize();
  cudaFree(d_nodes);
  heapify();
}

std::vector<PQNode> PriorityQueue::extract(int batch_size) {
  if (isEmpty()) {
    std::cout << "Priority queue is empty. No nodes were extracted."
              << std::endl;
    return std::vector<PQNode>();
  }
  std::vector<PQNode> h_nodes(batch_size);
  PQNode *d_nodes;
  cudaMalloc(&d_nodes, batch_size * sizeof(PQNode));

  int threadsPerBlock = 256;
  int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
  gpu_pq::extractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_pq, d_nodes,
                                                            batch_size, d_size);
  cudaDeviceSynchronize();
  cudaMemcpy(h_nodes.data(), d_nodes, batch_size * sizeof(PQNode),
             cudaMemcpyDeviceToHost);
  cudaFree(d_nodes);
  heapify();
  return h_nodes;
}

void PriorityQueue::heapify() {
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

int PriorityQueue::size() {
  cudaMemcpy(h_size, d_size, sizeof(int), cudaMemcpyDeviceToHost);
  return *h_size;
}

bool PriorityQueue::isEmpty() { return size() == 0; }
} // namespace gpu_pq
