#include "kernel.h"

__global__ void test_kernel(int* data) {
	int a = data[threadIdx.x];
	a++;
	data[threadIdx.x] = a;
}

void execute_kernel(int* data) {
	test_kernel<<<1, 64, 0>>>(data);

	cudaDeviceSynchronize();
}