#ifndef PBUFFER_H
#define PBUFFER_H

#include <vector>
#include <iostream>

class pBuffer {
public:
    // Constructor
    pBuffer(int capacity = 0) : size(0), capacity(capacity), array(capacity) {}

// Functions
    int get_size() { return size; }
    void insert(int value);

// private:
    int size;                  // Current size of the buffer
    int capacity;              // Maximum capacity of the buffer
    std::vector<int> array;    // Array holding the nodes
};


void pBuffer::insert(int value) {
    if (size == capacity) {
        std::cout << "Buffer is full, cannot insert more keys." << std::endl;
        return;
    }

    array[size] = value;
    size++;
    
}




#endif // PBUFFER_H
