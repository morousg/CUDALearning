#ifndef CUDAMEMPATTERNS_SRC_UTILS_H_
#define CUDAMEMPATTERNS_SRC_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

typedef unsigned int uint;

#define GLOBAL_ID ((blockIdx.x * blockDim.x) + threadIdx.x)

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

template <typename I1, typename I2=I1, typename O=I1>
struct binary_sum {
    __device__ O operator()(I1 input_1, I2 input_2) {return input_1 + input_2;}
};

template <typename I1, typename I2=I1, typename O=I1>
struct binary_mul {
    __device__ O operator()(I1 input_1, I2 input_2) {return input_1 * input_2;}
};

template <typename I1, typename I2=I1, typename O=I1>
struct binary_div {
    __device__ O operator()(I1 input_1, I2 input_2) {return input_1 / input_2;}
};

#endif //CUDAMEMPATTERNS_SRC_STANDARD_KERNELS_H_
