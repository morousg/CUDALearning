#include "kernel.h"
#include <math.h>

#define GLOBAL_ID ( (((blockDim.y * blockIdx.y) + threadIdx.y) * (gridDim.x * blockDim.x)) + ((blockDim.x * blockIdx.x) + threadIdx.x) )

__global__ void basic_copy_kernel(int* datain, int* dataout, int data_size) {
	int global_id = GLOBAL_ID;

	if (global_id < data_size)
		dataout[global_id] = datain[global_id];
}

// limited to data multiple of 4
__global__ void copy_kernel(int* datain, int* dataout, int data_size) {
	int global_id = GLOBAL_ID;

	if (global_id * 4 < data_size) {
		int4* datain_big = (int4*)datain;
		int4* dataout_big = (int4*)dataout;

		dataout[global_id] = datain[global_id];
	}
}

void execute_basic_copy_kernel(int* datain, int* dataout, int data_size, int block_size, cudaStream_t stream) {

	int num_blocks = (int)ceilf((float)data_size / (float)block_size);

	basic_copy_kernel<<<num_blocks, block_size, 0, stream>>>(datain, dataout, data_size);
}

void execute_copy_kernel(int* datain, int* dataout, int data_size, int block_size, cudaStream_t stream) {

	int num_blocks = (int)ceilf((float)data_size / (float)block_size / 4);

	copy_kernel<<<num_blocks, block_size, 0, stream >>>(datain, dataout, data_size);
}