#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "BPQ.h"

// k := the max batch size per "node" in the priority queue. This batch size is equivalent |cols|

#define rows 3
#define cols 3

int main() {
    

    BPQ test(3,3);

    int items1[3] = {1, 1, 1};  // Example row of items to insert
    int items2[3] = {2,2,2};
    int items3[3] = {3,3,3};
    test.insert(items1, 0);        // Insert into row 1
    test.insert(items2, 1);
    test.insert(items3, 2);

    test.test_print("1D");        // Print the updated array  

    test.swapRows(0,2);           // Swap rows 1 and 2

    test.test_print("1D");

    // test.test_insert();
    // std::cout << std::endl;
    // test.test_print("2D");
    // std::cout << std::endl;

    // // test.flatten();
    // test.test_print("1D");
    // std::cout << std::endl;

    // // std::cout << "Transferring to device" << std::endl;
    // test.transfer_to_device();
    
    // // std::cout << "Testing add" << std::endl;
    // test.test_add(1);

    // test.swapRows(0,1);
    

    
    return 0;
}





