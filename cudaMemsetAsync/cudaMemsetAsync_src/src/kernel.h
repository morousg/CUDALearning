#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

void launch_setAZero(uint* data, cudaStream_t stream);