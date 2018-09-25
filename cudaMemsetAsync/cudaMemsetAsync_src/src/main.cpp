#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>*/
#include "kernel.h"

int main() {

	uint *data;
	cudaMalloc(&data, sizeof(uint));

	cudaStream_t stream;
	cudaStreamCreate(&stream);

    launch_setAZero(data, stream);
    cudaMemsetAsync(data, 0, sizeof(uint), stream);

	cudaStreamSynchronize(stream);

    std::cout << "Executed!!" << std::endl;

	return 0;

}