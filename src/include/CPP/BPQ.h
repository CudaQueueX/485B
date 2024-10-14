// BPQ.h

#ifndef BPQ_H
#define BPQ_H

#include <vector>
#include <iostream>
#include <algorithm> // used for merge and sort

#include "Node.h"

#define MAX_SIZE 4
#define MAX_BUFFER_SIZE (MAX_SIZE - 1)


template <typename T>
class BPQ {
    public:
        // constructor, creates a BPQ object that has no elements in it 
        BPQ(int capacity, std::string test) {
            this->size = 0;
            this->capacity = capacity;
            this->test = test;
            insert_node(); // BPQ is empty at first, creating a new node.
        }
    
    // Function
    void infoDump();
    void insert_node();

    void create_and_insert(std::vector<int> values);
    
    void insert(std::vector<int> values);
    Node Extract_Min_Node();

    void HeapifyUp(int index);
    void HeapifyDown(int index);
    
    std::vector<Node> array; // Main array that holds the datastructure as a whole

    int size;   // the current size of the array
    int capacity; // max capacity of the array
    std::string test;

};

// Creates a new node inserts values and places it at the back of the array
template <typename T>
void BPQ<T>::create_and_insert(std::vector<int> values){
    Node temp(values.size()); // create a temp node
    for(int i = 0; i < values.size(); ++i) temp.insert(values[i]); // Insert the values

    sort(temp.array.begin(), temp.array.end()); // sort the values
    this->size++; // increment the size
    this->array.push_back(temp); // put it at the back
}


template <typename T>
void BPQ<T>::insert(std::vector<int> values) {
    // Attempt to insert the nodes into the root

    // Calculate if you can put the values into the root node 
    if (values.size() > array[0].spaceLeft()) {
        // std::cout << "Values cannot fit into the root node." << std::endl;
        // If the values cannot fit into the root node
        create_and_insert(values);

        // Call Heapify Up
        HeapifyUp(array.size() - 1);

        }
        else {
            // std::cout << "Values can fit into root node" << std::endl;
            // the values can fit into the root node
            for(int i = 0; i < values.size(); ++i) array[0].insert(values[i]);
            std::sort(array[0].array.begin(), array[0].array.end()); // sort the values to be in the correct order.
            }
}



template <typename T>
Node BPQ<T>::Extract_Min_Node() {
    // Extract the minimum node from the BPQ
    Node min = array[0]; // get the minimum node
    array[0] = array[size - 1]; // set the root node to the last node
    array.pop_back(); // remove the last node
    size--; // decrement the size
    HeapifyDown(0); // heapify down the tree
    return min; // return the minimum node
}

template <typename T>
void BPQ<T>::HeapifyUp(int index) {
    // Heapifying up the tree
    // std::cout << "Heapifying at index: " << index << std::endl;
    int parent = (index - 1) / 2;
    if (index > 0 && array[index].getMin() < array[parent].getMax()) {
        std::swap(array[index], array[parent]);
        HeapifyUp(parent);
    }
}



template <typename T>
void BPQ<T>::HeapifyDown(int index) {
    // Heapifying down the tree
    // std::cout << "Heapifying down at index: " << index << std::endl;
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    int smallest = index;

    if (left < size && array[left].getMin() < array[smallest].getMin())smallest = left;
    

    if (right < size && array[right].getMin() < array[smallest].getMin())smallest = right;
    

    if (smallest != index) {
        std::swap(array[index], array[smallest]);
        HeapifyDown(smallest);
    }
}



template <typename T>
void BPQ<T>::infoDump() {
    std::cout << std::endl;
    std::cout << "BPQ Info Dump:" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "\tSize: " << size << std::endl;
    std::cout << "\tCapacity: " << capacity << std::endl;
    std::cout << "\tTest: " << test << std::endl;

    // Iterate over each node and print its contents
    for (int i = 0; i < array.size(); ++i) {
        std::cout << "Node " << i << ":" << std::endl;
        array[i].infoDump();  // Use the Node's infoDump() to print its contents
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Inserts a new node into the BPQ vector of nodes using the current siz
template <typename T>
void BPQ<T>::insert_node() {
    // std::cout << "Creating a node" << std::endl;
    
    Node temp(4); // creating a node with a max capacity of 4

    array.push_back(temp); // vectors dont work like arrays, you need to use push_back to add new element.
    size++;

    // std::cout << "New node created, size is now:" << size << std::endl;
}






#endif // BPQ_H
