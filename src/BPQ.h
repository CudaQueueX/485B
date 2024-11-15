// BPQ.h



#ifndef BPQ_H
#define BPQ_H

#include <string>

class BPQ {
    public:
        BPQ(int rows, int cols);
        ~BPQ();

        // Testing
        void test_insert();
        void test_print(std::string type);
        void test_add(int value);
        
        // Getters
        int getSize();
        int get_current_pBuffer_size(int size);
        bool check_overFlow(int size);

        // Host
        void flatten();
        void swapRows(int row1, int row2);
        void transfer_to_device();
        void insert(int *items, int row);
        void insert_create_new_node();

        void bitonicSort(int *h_array, int size);
        bool isPowerOf2(int n);
        int nextPowerOf2(int n);
        int countTrailingZeros(int n);

        void increaseSize();


        bool BPQ_partial_insert(int *items, int size);
        void BPQ_insert(int *items, int size);
        void BPQ_sort_split(int *array1, int *array2, int *dest1, int *dest2, int size1, int size2, int Ma);
        void BPQ_sort_split2(int *Z, int Na, int *W, int Nb, int Ma);

        void BPQ_insert_heapify(int cur, int tar, int *items);


        void test_sort();
        




        

    private:
        int rows; // Number of rows of data in the array, used for chunking the 1D array into pieces
        int cols;
        int h_pBuffer_size;
        int h_node_count = 1;

        int *h_root_node; // Pointers to the start of the "root node" of the array on host
        int **h_array;  // 2D representation of the array on host 
        int *h_array_flat; // flattened version of the array on host
        int *h_pBuffer; // buffer node on host

        int *full_blocks;

        int *d_root_node; // Pointers to the start of the "root node" of the array on device
        int *d_array;   // 1D array on device
        int *d_pBuffer; // buffer node on device

};


// Kernel declarations
__global__ void addKernel(int *d_array, int rows, int cols);
__global__ void swapRowsKernel(int *d_array, int cols, int row1, int row2);
__global__ void insertKernel(int *d_array, int *items, int row, int cols);
__global__ void insert_pBuffer(int *d_array, int *d_pBuffer, int cols);

__global__ void bitonicSortKernel(int *d_array, int size, int pass, int stride);



#endif // BPQ_H

