#ifndef PBUFFER_H
#define PBUFFER_H

#include <vector>
#include <iostream>

class pBuffer {
public:
    // Constructor
    pBuffer(int capacity = 0) : size(0), capacity(capacity), array(capacity) {}

// Functionsz

private:
    int size;                  // Current size of the buffer
    int capacity;              // Maximum capacity of the buffer
    std::vector<int> array;    // Array holding the nodes
};






#endif // PBUFFER_H
