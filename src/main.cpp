// main.cpp

/**
 * Reference implementation of a MinHeap that is only run on the CPU.
 */


#include <iostream>
#include <vector>

#include "include/CPP/MinHeap.h"
#include "include/CPP/BPQ.h";

int main() {
    std::cout << "Mathew's Branch" << std::endl;
    std::cout << "Hello world" << std:: endl;

    // initalize the heap
    MinHeap<int> minHeap(6);
    std::vector<int> arr = {15,10,5,4,3,2};
    
    // Build the minheap
    std::cout << "Building Heap" << std::endl;
    minHeap.buildHeap(arr);
    std::cout << "Heap Built" << std::endl;

    std::cout << "Heap Contents" << std::endl;
    minHeap.printHeap();

    





    return 0;
}