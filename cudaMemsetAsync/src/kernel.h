#ifndef CUDAMEMSETASYNC_SRC_KERNEL_H_
#define CUDAMEMSETASYNC_SRC_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

typedef unsigned int uint;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void launch_setAZero(uint* data, cudaStream_t stream);

template <class T>
void launch_setNZeros(T* data, int num_elements, cudaStream_t stream);
#endif  // CUDAMEMSETASYNC_SRC_KERNEL_H_
