#ifndef CUDA_BASIC_PQ_H
#define CUDA_BASIC_PQ_H

#include <vector>

struct PQNode {
  int key;
  int value;
};

#define PQ_MAX_VAL PQNode{INT_MAX, 0}

namespace cuda_pq {

bool isPowerOf2(int n);
int nextPowerOf2(int n);

class PriorityQueue {
public:
  PriorityQueue(int capacity);
  ~PriorityQueue();

  void insert(std::vector<PQNode> h_nodes);
  void insert(PQNode h_node);
  std::vector<PQNode> extract(int batch_size);
  PQNode extract();
  int size() const;
  bool isEmpty() const;
  void print() const;

private:
  void swap_pq();
  void sort();
  int capacity;
  int *h_size;

  // Double buffer for the priority queue for speedup on extraction
  PQNode *d_pq;
  PQNode *d_pq_buffer;
  int *d_size;
};

} // namespace cuda_pq

#endif // CUDA_BASIC_PQ_H
