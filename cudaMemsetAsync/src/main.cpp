#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "./kernel.h"

int main() {
    uint *data;
    gpuErrchk(cudaMalloc(&data, sizeof(uint)));

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    launch_setAZero(data, stream);
    gpuErrchk(cudaMemsetAsync(data, 0, sizeof(uint), stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    std::cout << "Executed!!" << std::endl;

    return 0;
}
