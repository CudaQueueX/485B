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
        
        // Host
        void flatten();
        void swapRows(int row1, int row2);
        void transfer_to_device();
        void insert(int *items, int row);


        

    private:
        int rows;
        int cols;

        int *h_root_node; // Pointers to the start of the "root node" of the array on host
        int **h_array;  // 2D representation of the array on host 
        int *h_array_flat; // flattened version of the array on host
        int *h_pBuffer; // buffer node on host

        int *d_root_node; // Pointers to the start of the "root node" of the array on device
        int *d_array;   // 1D array on device
        int *d_pBuffer; // buffer node on device

};


// Kernel declarations
__global__ void addKernel(int *d_array, int rows, int cols);
__global__ void swapRowsKernel(int *d_array, int cols, int row1, int row2);
__global__ void insertKernel(int *d_array, int *items, int row, int cols);

#endif // BPQ_H

