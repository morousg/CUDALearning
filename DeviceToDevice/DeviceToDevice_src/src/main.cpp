#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.h"

int main() {

	int* data;
	cudaMalloc(&data, sizeof(int) * 64);

	execute_kernel(data);

    std::cout << "NICE" << std::endl;

	return 0;

}