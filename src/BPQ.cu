// BPQ.cu


/**
 * TODO:
 *  - Add Dynamic ressizing of arrays
 *  - Edit the bitonic sort to work with the updated version
 *  - Look into skip lists
 *  
 */


#include "BPQ.h"
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>


// Function Prototypes
__global__ void addKernel(int *d_array, int rows, int cols);
__global__ void swapRowsKernel(int *d_array, int cols, int row1, int row2);
__global__ void insertKernel(int *d_array, int *items, int row, int cols);
__global__ void insert_pBuffer(int *d_array, int *d_pBuffer, int cols);

__global__ void bitonicSortKernel(int *d_array, int size, int pass, int stride);

void bitonicSort(int *h_array, int size);
bool isPowerOf2(int n);
int nextPowerOf2(int n);
int countTrailingZeros(int n);

// Getters
int get_pBuffer_size(int size);
int getSize();


void insert(int *items, int row);
void flatten();
void swapRows(int row1, int row2);
void transfer_to_device();

bool BPQ_partial_insert(int *items, int size);
void BPQ_insert(int *items, int size);
void BPQ_sort_split(int *array1, int *array2, int *dest1, int *dest2, int size1, int size2, int Ma);

void test_add(int value);
void test_insert();
void test_print(std::string type);



__global__ void insert_pBuffer(int *d_array, int *d_pBuffer, int cols) {

}

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

__global__ void bitonicSortKernel(int *d_array, int size, int pass, int stride) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= size)
        return;

    unsigned int pairDistance = 1 << (stride - 1);
    unsigned int blockSize = 1 << pass;

    // Determine the indices of the elements to compare and swap
    unsigned int leftIdx = idx;
    unsigned int rightIdx = idx ^ pairDistance;

    if (rightIdx > leftIdx && rightIdx < size) {
        // Determine the direction of sorting (ascending or descending)
        bool ascending = ((idx / blockSize) % 2) == 0;

        // Perform the swap if needed
        if ((d_array[leftIdx] > d_array[rightIdx]) == ascending) {
            int temp = d_array[leftIdx];
            d_array[leftIdx] = d_array[rightIdx];
            d_array[rightIdx] = temp;
        }
    }
}


////////////////////////// KERNEL FUNCTIONS //////////////////////////

// Constructor
BPQ::BPQ(int rows, int cols) : rows(rows), cols(cols), h_pBuffer_size(0) {
    // Host allocation

    // h_array_flat = new int[rows * cols];
    h_array_flat = new int[1 * cols]; // Initially set to 1 row of data
    h_pBuffer = new int[cols - 1]; // pBuffer is of size cols - 1

    // Device Allocation
    cudaMalloc(&d_array, 1 * cols * sizeof(int));
    cudaMalloc(&d_pBuffer, cols - 1 * sizeof(int));

    h_root_node = h_array_flat; // So we can find the root node easily in the 1D array on host
    d_root_node = d_array; // So we can find the root node easily in the 1D array on device

    // Initializing the array to be full of -1's used to indicate empty positions
    for (int i = 0; i < 1 * cols; ++i) h_array_flat[i] = -1;
}

// Descrtuctor
BPQ::~BPQ() {
    std::cout << "Freeing memory" << std::endl;
    delete *h_array;
    delete h_pBuffer;
    // TODO: add error checking in here
    cudaFree(d_array);
    cudaFree(d_pBuffer);
}   

// This function is used to add another "node" to the BPQ
void BPQ::increaseSize() {
    std::cout << "Increasing size of the array" << std::endl;
    
    rows += 1;// Increase number of "rows"
    std::cout << "New size:" << rows << " " << rows * cols << std::endl;
    std::cout << std::endl; 

    // Increase array size on host
    int *temp = new int[rows * cols];
    for (int i = 0; i < (rows - 1) * cols; ++i) temp[i] = h_array_flat[i];

    // Set new row's contents to be all negative ones
    for (int i = (rows - 1) * cols; i < rows * cols; ++i) temp[i] = -1;

    free(h_array_flat);
    h_array_flat = temp;  // Update the pointer to the array

    // Increase array size on device
    int *d_temp;
    cudaMalloc(&d_temp, rows * cols * sizeof(int)); // Allocate new memory on device
    cudaMemcpy(d_temp, d_array, (rows - 1) * cols * sizeof(int), cudaMemcpyDeviceToDevice); // transfer the data over to the device
    
    // free the old memory
    cudaFree(d_array);
    d_array = d_temp; // Update the pointer to the array
}

void BPQ::test_sort() {
    std::cout << "Testing sort" << std::endl;
    bitonicSort(h_array_flat, rows * cols);
}

bool isPowerOf2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int nextPowerOf2(int n) {
    int count = 0;
    if (n && !(n & (n - 1)))
        return n;

    while (n != 0) {
        n >>= 1;
        count += 1;
    }

    return 1 << count;
}

// NEEDS TO BE UPDATED TO NOT REUSE GIANT SECTION OF CODE
void BPQ::bitonicSort(int *h_array, int size) {
    int oldSize = size;
    
    bool sizeUpdate = isPowerOf2(size);
    
    // TODO: FIX THE WHEN THE SIZE OF THE ARRAY ISNT POWER OF 2

    
    if(!sizeUpdate) {
        // SIZE OF ARRAY IS NOT A POWER OF TWO NEED TO UPDATE IT 
        std::cout << "Size of the array is not a power of 2" << std::endl;
        size = nextPowerOf2(size);

        // Creating temp location for this
        int *temp = new int[size];
        for (int i = 0; i < oldSize; ++i) temp[i] = h_array[i];
        for (int i = oldSize; i < size; ++i)temp[i] = 0;
        for (int i = 0; i < size; i++)std::cout << temp[i] << " ";
        std::cout << std::endl;

        // Allocate device memory
        int *d_array;
        cudaMalloc(&d_array, size * sizeof(int));
        cudaMemcpy(d_array, temp, size * sizeof(int), cudaMemcpyHostToDevice); 

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        int numPasses = countTrailingZeros(size);
        for(int pass = 1; pass <= numPasses; ++pass) {
            for (int stride = pass; stride > 0; --stride) {
                bitonicSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, size, pass, stride);
                cudaDeviceSynchronize();
            }
        }

        cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_array);

        // Restore the original size of the array by trimming off the extra added elements
        for (int i = size - oldSize; i < size; i++) h_array[i - size + oldSize] = h_array[i];
        
        
        for (int i = 0; i < oldSize; i++)std::cout << h_array[i] << " ";
        std::cout << std::endl;

    }
    else {
        // Allocate device memory

        std::cout << "Size of the array is a power of 2" << std::endl;
        int *d_array;
        cudaMalloc(&d_array, size * sizeof(int));
        cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice); 

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        int numPasses = countTrailingZeros(size);
        for(int pass = 1; pass <= numPasses; ++pass) {
            for (int stride = pass; stride > 0; --stride) {
                bitonicSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, size, pass, stride);
                cudaDeviceSynchronize();
            }
        }

        cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_array);
    
        for (int i = 0; i < size; i++) std::cout << h_array[i] << " ";
        std::cout << std::endl;
    }
}


void BPQ::BPQ_insert(int *items, int size) {
    // Assumption that the array is already on the host with the most updated version

    // Sort the incoming items
    bitonicSort(items, size);

    // Attempt a partial insert into root/buffer
    if(BPQ_partial_insert(items, size)) {
        std::cout << "Returned True" << std::endl;
        return;
        }
    

}

void BPQ::BPQ_insert_heapify(int cur, int tar, int *items) {

}

bool BPQ::BPQ_partial_insert(int *items, int size) {
    
    std::cout << "Attempting partial insert" << std::endl;
    // If heap is empty
    if(h_array_flat[0] == -1) {
        // -1 => that the heap is empty
        
        std::cout << "Heap has no elements in it, inserting into the root node" << std::endl;
        
        // Insert into root node
        insert(items, 0); // inserts into root node
        return true;
    }

    // SORT SPLIT OCCURS HERE
    



    // Check if the buffer is empty
    if(!check_overFlow(size)) {
        // Does not cause overflow => insert into buffer

        // Copy the items into the buffer
        for (int i = 0; i < size; ++i) h_pBuffer[i] = items[i];
        h_pBuffer_size = size;

        for (int i = 0; i < size; i++) std::cout << h_pBuffer[i] << " ";
        std::cout << std::endl;
        return true;


    }
    else {
        // Causes overflow; need to handle the buffer overflow
        std::cout << "Causes overflow" << std::endl;

        // Sort the pBuffer
        bitonicSort(h_pBuffer, cols - 1);
        return false;
    }


} 

bool BPQ::check_overFlow(int size) {
// Checks if inserting items will cause overflow

    // True => overflow
    // False => no overflow
    if(h_pBuffer_size == 0) return false; // No items are in the buffer

    if(h_pBuffer_size + size > cols - 1) return true; // Items are in the buffer

    // default to no
    return false;
}

int BPQ::getSize() {
    // Under assumption that the array is in the host not the device
    
    for(int i = 0; i < cols; ++i) {
        if(h_array_flat[i] == -1) {
        }
    }

    return 0;
}

int BPQ::get_current_pBuffer_size(int size) {
    int retval = 0;
    for (int i = 0; i < cols -1; ++i) {
        if (h_pBuffer[i] != -1) retval++;   
    }
    return retval;
}

// To mimic the __builtin_ctz() function in C++, wasn't working properly 
int BPQ::countTrailingZeros(int n) {
    if (n == 0) return 32;  // Assuming 32-bit integers; all bits are zero.
    int count = 0;
    while ((n & 1) == 0) {  // Check the least significant bit.
        n >>= 1;            // Right shift by 1 (divide by 2).
        count++;
    }
    return count;
}

bool BPQ::isPowerOf2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int BPQ::nextPowerOf2(int n) {
    int count = 0;
    if(n && !(n & (n - 1))) return n;

    while(n != 0) {
        n >>= 1;
        count += 1;
    }

    return 1 << count;
}

void BPQ::insert(int *items, int row) {

    bitonicSort(items, cols);

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
    cudaDeviceSynchronize();
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