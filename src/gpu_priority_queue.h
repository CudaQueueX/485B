#ifndef GPU_PRIORITY_QUEUE_H
#define GPU_PRIORITY_QUEUE_H

#include <vector>

struct PQNode {
  int key;
  int value;
};

#define PQ_MAX_VAL PQNode{INT_MAX, 0}

namespace gpu_pq {

bool isPowerOf2(int n);
int nextPowerOf2(int n);

class PriorityQueue {
public:
  PriorityQueue(int capacity);
  ~PriorityQueue();

  void insert(std::vector<PQNode> h_nodes);
  std::vector<PQNode> extract(int batch_size);
  void heapify();
  int size();
  bool isEmpty();
  void print();

private:
  int capacity;
  int *h_size;

  PQNode *d_pq;
  int *d_size;
};

} // namespace gpu_pq

#endif // GPU_PRIORITY_QUEUE_H
