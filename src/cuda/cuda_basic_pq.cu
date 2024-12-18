#include "cuda_basic_pq.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits.h>

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

void PriorityQueue::sort_bitonic() {
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
__global__ void findMaxKernel(PQNode *d_pq, int *d_maxKey, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory to store the local max in each block
    __shared__ int localMax[256]; 
    // Initialize shared memory
    if (idx < size)
    localMax[threadIdx.x] = d_pq[idx].key;
    else localMax[threadIdx.x] = INT_MIN; // Handle out-of-bounds threads
    
    __syncthreads();

    // Reduce within the block to find the local maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && idx + stride < size)localMax[threadIdx.x] = max(localMax[threadIdx.x], localMax[threadIdx.x + stride]);
        __syncthreads();
    }

    // The first thread in the block writes its result to global memory
    if (threadIdx.x == 0) atomicMax(d_maxKey, localMax[0]);
    
}



void PriorityQueue::sort_radix() {
  int size = this->size();

  int *d_key;
  cudaMalloc(&d_key, size * sizeof(int));
  cudaMemset(d_key, INT_MIN, sizeof(int));

  const int maxThreadsPerBlock = 256;
  int threadsPerBlock = std::min(size, maxThreadsPerBlock);
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  findMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_pq, d_key, size);
  cudaDeviceSynchronize();

  int max_key;
  cudaMemcpy(&max_key, d_key, sizeof(int), cudaMemcpyDeviceToHost);

  PQNode *d_temp;
  cudaMalloc(&d_temp, size * sizeof(PQNode));


  // Radix sort by each digit
  for (int exp = 1; max_key / exp > 0; exp *= 10) {
      countingSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_pq, d_temp, nullptr, size, exp);
      cudaDeviceSynchronize();

      // Swap buffers
      PQNode *temp = d_pq;
      d_pq = d_temp;
      d_temp = temp;
  }

  // Clean up temporary buffer
  cudaFree(d_temp);

}



// Using the C++ version of counting sort which is changed to run with cuda
__global__ void countingSortKernel(PQNode *d_input, PQNode *d_output, int *d_count, int n, int exp) {
  // idx is used per thread to select its specific digit.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;


  if(idx < n) {
    // create shared memory for the count array
    __shared__ int count[10];
    if(threadIdx.x < 10) count[threadIdx.x] = 0; // initialize the counts to be 0
    __syncthreads();

    int temp = (d_input[idx].key / exp) % 10;
    atomicAdd(&count[temp], 1); // increment the count of the digit
    __syncthreads();

    // calculate the prefix sum of the count array
    if(threadIdx.x < 10) {
      for(int i = 1; i <= threadIdx.x; i++) {
        count[threadIdx.x] += count[threadIdx.x - i];
      }
    }
    __syncthreads();

    // output
    if(idx < n) {
      d_output[count[temp]] = d_input[idx];
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
