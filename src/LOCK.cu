

#include "LOCK.h"
#include <cuda.h>
#include <cuda_runtime.h>


__device__ void LOCK::lock() {
    while (atomicCAS(&mutex, 0, 1) != 0);
}

__device__ void LOCK::unlock() {
    atomicExch(&mutex, 0);
}