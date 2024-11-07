// BPQ.cu


#include "BPQ.h"
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>



__global__ void addKernel(int *d_array, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows * cols) {
        atomicAdd(&d_array[idx], 1);
    }
}


__global__ void swapRowsKernel(int *d_array, int cols, int row1, int row2) {
    int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure colIdx is within bounds of the row's column count
    if (colIdx < cols) {
        // Calculate the starting indices of each row in the 1D array
        int idx1 = row1 * cols + colIdx;
        int idx2 = row2 * cols + colIdx;

        // Swap elements between row1 and row2
        int temp = d_array[idx1];
        d_array[idx1] = d_array[idx2];
        d_array[idx2] = temp;
    }
}


__global__ void insertKernel(int *d_array, int *items, int row, int cols) {
    int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (colIdx < cols) {
        int idx = row * cols + colIdx;  // Calculate the index in the 1D array
        d_array[idx] = items[colIdx];    // Insert item into d_array at the correct position
    }
}

////////////////////////// KERNEL FUNCTIONS //////////////////////////


BPQ::BPQ(int rows, int cols) : rows(rows), cols(cols) {
    // Host allocation
    h_array = new int*[rows]; // Alocating memory for the array of pointers
    for (int i = 0; i < rows; i++) h_array[i] = new int[cols]; // Allocating memory for each of the rows
    h_pBuffer = new int[cols - 1]; // pBuffer is of size cols - 1
    h_array_flat = new int[rows * cols];


    // Device Allocation
    cudaMalloc(&d_array, rows * cols * sizeof(int));
    cudaMalloc(&d_pBuffer, cols - 1 * sizeof(int));

    h_root_node = h_array_flat; // So we can find the root node easily in the 1D array on host
    d_root_node = d_array; // So we can find the root node easily in the 1D array on device


    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            h_array[i][j] = -1;
        }
    }

    for (int i = 0; i < rows * cols; ++i) h_array_flat[i] = -1;
    

    
}


BPQ::~BPQ() {
    std::cout << "Freeing memory" << std::endl;
    delete *h_array;
    delete h_pBuffer;
    // TODO: add error checking in here
    cudaFree(d_array);
    cudaFree(d_pBuffer);
}   



void BPQ::insert(int *items, int row) {
    if (row >= rows || row < 0) {
        std::cerr << "Invalid row index for insertion." << std::endl;
        return;
    }

    int *d_items;  // Device array for the items to insert
    cudaMalloc(&d_items, cols * sizeof(int));

    // Copy items to device memory
    cudaMemcpy(d_items, items, cols * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to insert items into the specified row
    int threadsPerBlock = 256;  
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;

    insertKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, d_items, row, cols);

    // Synchronize to ensure insertion is complete before continuing
    cudaDeviceSynchronize();

    // Free device memory for items
    cudaFree(d_items);

    // Optionally update host array if needed
    cudaMemcpy(h_array_flat, d_array, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
}


void BPQ::flatten() {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            h_array_flat[i * cols + j] = h_array[i][j];
        }
    }
}



void BPQ::swapRows(int row1, int row2) {
    if (row1 >= rows || row2 >= rows || row1 < 0 || row2 < 0) {
        std::cerr << "Invalid row indices for swapping." << std::endl;
        return;
    }
    
    // test_print("1D");

    int threadsPerBlock = 256;  // Number of threads per block
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;  // Calculate number of blocks

    // Launch the kernel to swap rows in d_array
    swapRowsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, cols, row1, row2);
    
    // Synchronize to make sure swap is completed
    cudaDeviceSynchronize();

    // Copy the modified d_array back to h_array_flat
    cudaMemcpy(h_array_flat, d_array, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    // test_print("1D");
}



void BPQ::transfer_to_device() {
    cudaMemcpy(d_array, h_array_flat, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
}


////////////////////////// TESTING FUNCTIONS //////////////////////////
void BPQ::test_add(int value){
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows * cols + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, rows, cols);
    cudaMemcpy(h_array_flat, d_array, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    // test_print("1D");
}


void BPQ::test_insert() {

    int counter = 0;

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            h_array[i][j] = counter;
            counter++;
        }
    }

}

void BPQ::test_print(std::string type) {
    if (type == "1D") {
        for (int i = 0; i < rows * cols; ++i) {
            std::cout << h_array_flat[i] << " ";
        }
        std::cout << std::endl;
    }
    else if  (type == "2D") {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << h_array[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        std::cout << "Invalid type" << std::endl;
    }
}