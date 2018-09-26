#include "./kernel.h"
#include <math.h>

__global__ void kernel_setAZero(uint* data) {
    *data = 0u;
}

void launch_setAZero(uint* data, cudaStream_t stream) {
    kernel_setAZero <<<1, 1, 0, stream>>>(data);
    gpuErrchk(cudaGetLastError());
}
