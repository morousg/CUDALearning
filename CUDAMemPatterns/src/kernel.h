#ifndef CUDAMEMPATTERNS_SRC_KERNEL_H_
#define CUDAMEMPATTERNS_SRC_KERNEL_H_

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

template <typename Operator, typename I1, typename I2, typename O>
struct _binary_operation_scalar {
    I2 scalar;
    Operator nv_operator;
};

template <typename Operator, typename I1, typename I2=I1, typename O=I1>
using binary_operation_scalar = typename _binary_operation_scalar<Operator, I1, I2, O>;

template <typename Operator, typename I1, typename I2, typename O>
struct _binary_operation_pointer {
    I2* pointer;
    Operator nv_operator;
    I2 temp_register[4];
};

template <typename Operator, typename I1, typename I2=I1, typename O=I1>
using binary_operation_pointer = typename _binary_operation_pointer<Operator, I1, I2, O>;

void test2_mult_sum_div_float(float* data, dim3 data_dims, cudaStream_t stream);

#endif  // CUDAMEMPATTERNS_SRC_KERNEL_H_
