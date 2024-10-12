#ifndef NODE_H
#define NODE_H

#include <vector>
#include <iostream>
#include <random>

class Node {
    public:
        // Constructor
        Node(int capacity = 0) : size(0), capacity(capacity), array(capacity) {}

    // Functions
    void test_insert(int temp);
    void infoDump();
    void insert(int value);    



        int size;                  // Current number of elements within the node
        int capacity;              // Maximum capacity of the node
        std::vector<int> array;    // Array containing the node data in the form of integers
    
    private:
        // int size;                  // Current number of elements within the node
        // int capacity;              // Maximum capacity of the node
        // std::vector<int> array;    // Array containing the node data in the form of integers
};

void Node::insert(int value) {
    if (size == capacity) {
        std::cout << "Node is full, cannot insert more keys." << std::endl;
        return;
    }

    array[size] = value;
    size++;
}

void Node::infoDump() {
    std::cout << "Size: " << size << std::endl;
    std::cout << "Capacity: " << capacity << std::endl;
    std::cout << "Array: ";
    for (int i = 0; i < capacity; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void Node::test_insert(int temp){
    
    // Seed with a real random value, if available
    std::random_device rd;

    // Initialize random engine and distribution
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine
    std::uniform_int_distribution<> dis(1, 100);  // Range [1, 100]

    for (int i = 0; i < 3; i++) {
        // Generate and print a random number
        int randomNumber = dis(gen);
        std::cout << randomNumber << std::endl;

        array[i] = randomNumber;
        size++;
    }

}

#endif // NODE_H
