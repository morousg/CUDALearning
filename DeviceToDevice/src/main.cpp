#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.h"

#define DATA_SIZE 3840*2160
#define BLOCK_SIZE 512

int main() {

	int *datain, *dataout, *dataout2;
	cudaMalloc(&datain, sizeof(int) * DATA_SIZE);
	cudaMalloc(&dataout, sizeof(int) * DATA_SIZE);
	cudaMalloc(&dataout2, sizeof(int) * DATA_SIZE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	execute_basic_copy_kernel(datain, dataout, DATA_SIZE, BLOCK_SIZE, stream);
	execute_copy_kernel(datain, dataout, DATA_SIZE, BLOCK_SIZE, stream);
	cudaMemcpyAsync(dataout2, datain, DATA_SIZE, cudaMemcpyDeviceToDevice, stream);

	cudaStreamSynchronize(stream);

    std::cout << "Executed!!" << std::endl;

	return 0;

}