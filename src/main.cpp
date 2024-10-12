// main.cpp

/**
 * Reference implementation of a MinHeap that is only run on the CPU.
 */


#include <iostream>
#include <vector>
#include <random>

#include "include/CPP/BPQ.h"


std::vector<int> gen_vector(int m);

int main() {

    // std::cout << "Mathew's Branch" << std::endl;

    BPQ<int> bpq(4, "testing");


    // Here

    std::vector<int> temp = gen_vector(4);
    std::vector<int> temp2 = gen_vector(3); 

    bpq.insert_val(temp);
    bpq.insert_val(temp2);

    bpq.infoDump();
    // std::vector<int> temp2 = gen_vector(4);



    return 0;
}





//Function that generates vectors of size m, fills them with random values and returns them
std::vector<int> gen_vector(int m) {
    std::vector<int> retval;

    // Seed with a real random value, if available
    std::random_device rd;

    // Initialize random engine and distribution
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine
    std::uniform_int_distribution<> dis(1, 100);  // Range [1, 100]

    for(int i = 0; i < m; ++i)retval.push_back(dis(gen));

    return retval;
}