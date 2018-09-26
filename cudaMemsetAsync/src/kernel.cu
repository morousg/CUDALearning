#include "./kernel.h"
#include <math.h>

#define GLOBAL_ID ((blockIdx.x * blockDim.x) + threadIdx.x)

__global__ void kernel_setAZero(uint* data) {
    *data = 0u;
}

template <class T, T zero>
__global__ void kernel_setNZeros(T* data, int num_elements) {
    if (GLOBAL_ID < num_elements)
        data[GLOBAL_ID] = zero;
}

__global__ void kernel_steNZeros4(uint* data, int num_4elements) {
    if (GLOBAL_ID < num_4elements) {
        uint4* data4 = (uint4*)data;
        uint4 zeros = { 0u, 0u, 0u, 0u };
        data4[GLOBAL_ID] = zeros;
    }
}

void launch_setAZero(uint* data, cudaStream_t stream) {
    kernel_setAZero <<<1, 1, 0, stream>>>(data);
    gpuErrchk(cudaGetLastError());
}

template <class T>
void launch_setNZeros(T* data, int num_elements, cudaStream_t stream) {
    int blockSize = 0;
    int gridSize = 0;
    bool four_elem = false;
    if (num_elements % 4 == 0) {
        num_elements /= 4;
        four_elem = true;
    }
    if (num_elements <= 512) {
        blockSize = num_elements;
        gridSize = 1;
    } else {
        blockSize = 512;
        gridSize = (int)ceilf((float)num_elements / 512.);
    }

    if (four_elem) {
        kernel_steNZeros4<< <gridSize, blockSize, 0, stream >> > (data, num_elements);
        gpuErrchk(cudaGetLastError());
    } else {
        kernel_setNZeros<T, (T)0> << <gridSize, blockSize, 0, stream >> > (data, num_elements);
        gpuErrchk(cudaGetLastError());
    }
}

template
void launch_setNZeros<uint>(uint* data, int num_elements, cudaStream_t stream);
/*template
void launch_setNZeros<char>(char* data, int num_elements, cudaStream_t stream);*/
