#ifndef MINHEAP_H
#define MINHEAP_H

#include <vector>
#include <iostream>


template <typename T>
class MinHeap {
    public:
        // constructor
        MinHeap(int capacity) {
            this->size = 0;
            this->capacity = capacity;
            this->array.resize(capacity);
        }

        T peek();
        T ExtractMin();
        void Heapify(int i);
        void buildHeap(const std::vector<T>& v);
        void insertNode(T value);
        void deleteNode(T key);
        void printHeap();

    private:
        std::vector<T> array;
        int size;
        int capacity;
};

// Peeks the top of the heap
template <typename T>
T MinHeap<T>::peek() {
    if (size <= 0) return -1;
    return array[0];
}

// Will extract the minimum
template <typename T>
T MinHeap<T>::ExtractMin() {
    if (size <= 0) return -1;
    if (size == 1) return array[0];

    T root = array[0];
    array[0] = array[size - 1];
    size--;
    Heapify(0);
    return root;
}

template <typename T>
void MinHeap<T>::Heapify(int i) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < size && array[left] < array[smallest]) smallest = left;
    if (right < size && array[right] < array[smallest]) smallest = right;
    if (smallest != i) {
        std::swap(array[i], array[smallest]);
        Heapify(smallest);
    }
}

template <typename T>
void MinHeap<T>::buildHeap(const std::vector<T>& v) {
    capacity = v.size();
    size = capacity;
    array = v;

    for (int i = (size / 2) - 1; i >= 0; --i) Heapify(i);
}

template <typename T>
void MinHeap<T>::insertNode(T value) {
    if (size == capacity) {
        capacity *= 2;
        array.resize(capacity);
    }

    size++;
    int i = size - 1;
    array[i] = value;
    while (i != 0 && array[(i - 2) / 2] > array[i]) {
        std::swap(array[i], array[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

template <typename T>
void MinHeap<T>::deleteNode(T key) {
    int index = -1;
    for (int i = 0; i < size; i++) {
        if (array[i] == key) {
            index = i;
            break;
        }
    }

    if (index == -1) {
        std::cout << "Key not found" << std::endl;
        return;
    }

    if (index == size - 1) {
        size--;
        return;
    }

    array[index] = array[size - 1];
    size--;
    Heapify(index);
}

template <typename T>
void MinHeap<T>::printHeap() {
    int level = 0;
    int elementsInLevel = 1;
    int count = 0;

    for (int i = 0; i < size; ++i) {
        if (count == elementsInLevel) {
            std::cout << std::endl;
            level++;
            elementsInLevel = 1 << level;  // 2^level
            count = 0;
        }
        std::cout << array[i] << " ";
        count++;
    }
    std::cout << std::endl;
}

#endif // MINHEAP_H
