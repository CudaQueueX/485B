// BPQ.h

#ifndef BPQ_H
#define BPQ_H

#include <vector>
#include <iostream>
#include <algorithm> // used for merge and sort

#include "Node.h"
#include "pBuffer.h"

#define MAX_SIZE 4
#define MAX_BUFFER_SIZE (MAX_SIZE - 1)


template <typename T>
class BPQ {
    public:
        // constructor, creates a BPQ object that has no elements in it 
        BPQ(int capacity, std::string test) {
            this->size = 0;
            this->capacity = capacity;
            this->Buffer = pBuffer(MAX_BUFFER_SIZE);
            this->test = test;
            insert_node(); // BPQ is empty at first, creating a new node.
        }
    
    // Function
    void infoDump();
    void insert_node();
    
    void insert_val(std::vector<int> values);
    void insert_heapify();
    void partial_insert(std::vector<int> items);
    void sort_split(std::vector<int> items, Node &root);



    
    std::vector<Node> array; // Main array that holds the datastructure as a whole
    pBuffer Buffer; // Buffer array
    int size;   // the current size of the array
    int capacity; // max capacity of the array
    std::string test;

    private:
        // std::vector<Node> array; // Main array that holds the datastructure as a whole
        // pBuffer Buffer; // Buffer array
        // int size;   // the current size of the array
        // int capacity; // max capacity of the array
        // std::string test;

};




template <typename T>
void BPQ<T>::insert_val(std::vector<int> values) {
    std::sort(values.begin(), values.end());  // Sort incoming values.

    // If the heap is empty, create a new node.
    if (size == 0) {
        std::cout << "Heap was empty, creating a new node and inserting values." << std::endl;
        insert_node();
    }

    // Check if the root node is empty.
    if (array[0].size == 0) {
        for (int val : values) {
            array[0].insert(val);  // Insert values directly if the root is empty.
        }
    } else {
        // Call sort_split if the root node is not empty.
        sort_split(values, array[0]);
    }
}



/**
 * Called to insert keys, either into the buffer or as a new heap node. 
 * 
 * If the heap is empty, inserted items are directly placed into the root
 * If the heap is not empty a sort_split is called between the root node and "items" to place the smaller keys in the root node
 *      we then check if placing all insert keys will overflow the buffer
 * 
 *          If not the updated "items" are inserted into the buffer
 */

template <typename T>
void BPQ<T>::partial_insert(std::vector<int> items) {

    

    // Keys are inserted into buffer or into a new node.
    if(this->Buffer.get_size() == 0) {
        // Buffer is empty, check if we can insert all the keys into the buffer
        
        if (this->Buffer.size + items.size() > MAX_BUFFER_SIZE) {
            // input vector would make the buffer overflow
            std::cout << "Inserting the keys would overflow the buffer." << std::endl;
            }
            else {
                // Insert the keys into the buffer at the correct index, not overwriting anything
                for(int i = 0; i < items.size(); ++i){ this->Buffer.insert(items[i]); }
                
            }

    }



}

template <typename T>
void BPQ<T>::sort_split(std::vector<int> items, Node& root) {
    std::cout << "Performing sort_split..." << std::endl;

    // Step 1: Merge root's elements and incoming items into a single vector.
    std::vector<int> merged = root.array;  // Start with root node's array.
    merged.resize(root.size);  // Resize to match the current number of elements.
    merged.insert(merged.end(), items.begin(), items.end());  // Add incoming items.

    // Step 2: Sort the merged array.
    std::sort(merged.begin(), merged.end());

    // Step 3: Extract the smallest 'root.capacity' elements for the root node.
    int newRootSize = std::min(static_cast<int>(merged.size()), root.capacity);
    for (int i = 0; i < newRootSize; ++i) {
        root.array[i] = merged[i];  // Place the smallest elements in the root.
    }
    root.size = newRootSize;  // Update root's size.

    // Step 4: Handle the remaining elements (if any).
    std::vector<int> remaining(merged.begin() + newRootSize, merged.end());

    if (!remaining.empty()) {
        // Check if remaining elements fit into the buffer.
        if (Buffer.size + remaining.size() <= Buffer.capacity) {
            for (int val : remaining) {
                Buffer.insert(val);  // Insert into the buffer.
            }
        } else {
            std::cout << "Warning: Not enough space in buffer to store remaining elements." << std::endl;
        }
    }

    std::cout << "sort_split completed." << std::endl;
}


template <typename T>
void BPQ<T>::infoDump() {
    std::cout << "BPQ Info Dump:" << std::endl;
    std::cout << "Size: " << size << std::endl;
    std::cout << "Capacity: " << capacity << std::endl;
    std::cout << "Test: " << test << std::endl;

    // Iterate over each node and print its contents
    for (int i = 0; i < array.size(); ++i) {
        std::cout << "Node " << i << ":" << std::endl;
        array[i].infoDump();  // Use the Node's infoDump() to print its contents
    }
}


// Inserts a new node into the BPQ vector of nodes using the current siz
template <typename T>
void BPQ<T>::insert_node() {
    std::cout << "Creating a node" << std::endl;
    
    Node temp(4);

    array.push_back(temp); // vectors dont work like arrays, you need to use push_back to add new element.
    size++;

    std::cout << "New node created, size is now:" << size << std::endl;
}



#endif // BPQ_H
