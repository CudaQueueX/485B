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
    void partial_insert();
    void sort_split();



    
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
    
    std::sort(values.begin(), values.end()); // sort the incoming values in ascending order


    // Create a node if the heap is empty
    if(size == 0) {
        // Create a node, should never get here because the constructor creates a node
        std::cout << "Heap was empty, creating a new node and inserting it" << std::endl;
        insert_node();
    }

    // Insert values into the heap
    if (bpq.array[0].size == 0) {
        // The root node is empty, insert the values into the root node
        

    
        // Heap is not empty

        // Merge m keys with the root then sort them

        // Insert the largest m keys into the pBuffer

            // check if pBuffer overflows

                // No => sort the pbuffer & done

                // yes => 
    }

    


        // YES => just insert the keys into the root

        // No => More steps
        // Merge m keys with the root >> insert largest M keys into the pBuffer >> check if it overflows >> 
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
void BPQ<T>::partial_insert() {

}

template <typename T>
void BPQ<T>::infoDump() {
    std::cout << "Size: " << size << std::endl;
    std::cout << "Capacity: " << capacity << std::endl;
    std::cout << "Test: " << test << std::endl;
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
