#include <cuda.h>
#include <cuda_runtime.h>

void execute_basic_copy_kernel(int* datain, int* dataout, int data_size, int block_size, cudaStream_t stream);
void execute_copy_kernel(int* datain, int* dataout, int data_size, int block_size, cudaStream_t stream);