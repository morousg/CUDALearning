#ifndef CUDAMEMPATTERNS_SRC_KERNEL_H_
#define CUDAMEMPATTERNS_SRC_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvfunctional>

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

template <typename I1, typename I2, typename O>
__host__ __device__ O binary_sum(I1 input_1, I2 input_2) {
    return input_1 + input_2;
}

enum transform_patern {
    scalar,
    pointer
};

template <typename I1, typename I2, typename O>
struct _binary_operation {
    transform_patern parameter;
    nvstd::function<O(I1,I2)> nv_operator;
    I2 scalar;
    I2* pointer;
    I2 temp_register[4];
};

template <typename I1, typename I2, typename O>
using binary_operation = typename _binary_operation<I1, I2, O>;

void test_mult_sum_div_float(float* data, float mul_val, float* sum_array, float div_val,
                             dim3 data_dims, cudaStream_t stream);

#endif  // CUDAMEMPATTERNS_SRC_KERNEL_H_
