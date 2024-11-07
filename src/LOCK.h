// LOCK.h

#ifndef LOCK_H
#define LOCK_H

#include <cuda_runtime.h>

class LOCK {
    public:
        __device__ void lock();
        __device__ void unlock();

    private:
        int mutex = 0;
};

#endif // LOCK_H