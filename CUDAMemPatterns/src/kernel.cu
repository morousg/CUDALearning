#include "./kernel.h"

#define GLOBAL_ID ((blockIdx.x * blockDim.x) + threadIdx.x)

void test_mult_sum_div_float(float* data, float mul_val, float* sum_array, float div_val,
                             dim3 data_dims, cudaStream_t stream) {
    // We don't think about step or ROI's yet.
    dim3 thread_block(512);
    dim3 grid((data_dims.x * data_dims.y) / 512);
}
