#ifndef REF_BASIC_PQ_H
#define REF_BASIC_PQ_H

#include <vector>
#include "../node.h"

struct PQNode {
  Node node; // encapsulation of the Node class
  
  PQNode(const Node &n) : node(n) {} // constructor 
  
  // default constructor
  PQNode() : node(Node(-1, -1, false)) {}
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

  void insert(const Node& n); // overloaded insert function
  Node extractNode(); // updated extract node function

  std::vector<PQNode> extract(int batch_size);
  PQNode extract();
  int size() const;
  bool isEmpty() const;
  void print() const;

private:
  // Standard library sort
  void sort();
  // Merge Sort
  void merge(std::vector<int>& arr, int left, int mid, int right);
  void merge_sort(std::vector<int>&arr, int left, int right);
  // Radix sort
  void radix_sort(std::vector<int>& arr);
  void counting_sort(std::vector<int>& input, int exp);

  std::vector<PQNode> pq;
  int capacity;
  int pq_size;
};

} // namespace ref_pq

#endif // REF_BASIC_PQ_H
