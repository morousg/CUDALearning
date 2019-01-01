#include <vector>
#include "./kernel.h"
#include "cpu_baseline.h"

int main() {

    float* data = (float*)malloc(sizeof(float)*SIZE);
    float* h_data = (float*)malloc(sizeof(float)*SIZE);

    for (int i=0; i<SIZE; ++i) {
        data[i] = (float)i;
    }

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    float* d_data;
    gpuErrchk(cudaMalloc(&d_data, sizeof(float)*SIZE));
    gpuErrchk(cudaMemcpyAsync(d_data, data, sizeof(float)*SIZE, cudaMemcpyHostToDevice, stream));

    dim3 size(128);
    test2_mult_sum_div_float(d_data, size, stream);

    gpuErrchk(cudaMemcpyAsync(h_data, d_data, sizeof(float)*SIZE, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    cpu_binary_operation<float> op1 = {scalar, cpu_binary_mul<float>, 5};
    cpu_binary_operation<float> op2 = {pointer, cpu_binary_sum<float>, 0, data};
    cpu_binary_operation<float> op3 = {scalar, cpu_binary_div<float>, 2};

    cpu_cuda_transform(data, data, op1, op2, op3);

    std::cout << "Executed!!" << std::endl;
    bool success = true;
    for (int i=0; i<SIZE; ++i) {
        if (success) success &= 0.001 > data[i] - h_data[i];
        std::cout << "data " << data[i] << " == " << h_data[i] << " h_data" << std::endl;
    }
    free(data);

    if (success) {
        std::cout << "Success!!" << std::endl;
    } else {
        std::cout << "Fail!!" << std::endl;
    }

    return 0;
}
