#ifndef REF_BASIC_PQ_H
#define REF_BASIC_PQ_H

#include <vector>

struct PQNode {
  int key;
  int value;
};

namespace ref_pq {

/*
This is a reference implementation of a priority queue that uses a sorted array
to store elements.
*/
class PriorityQueue {
public:
  PriorityQueue(int capacity);
  ~PriorityQueue();

  void insert(std::vector<PQNode> nodes);
  void insert(PQNode node);
  std::vector<PQNode> extract(int batch_size);
  PQNode extract();
  int size() const;
  bool isEmpty() const;
  void print() const;

private:
  void sort();

  std::vector<PQNode> pq;
  int capacity;
  int pq_size;
};

} // namespace ref_pq

#endif // REF_BASIC_PQ_H
