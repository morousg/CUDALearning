#include "./kernel.h"

#define GLOBAL_ID ((blockIdx.x * blockDim.x) + threadIdx.x)

__global__ void kernel_setAZero(uint* data) {
    *data = 0u;
}

template <class T, T zero>
__global__ void kernel_setNZeros(T* data, int num_elements) {
    if (GLOBAL_ID < num_elements)
        data[GLOBAL_ID] = zero;
}

__global__ void kernel_steNZeros4(uint* data, int num_elements, int num_4elems) {
    if (num_4elems * 4 == num_elements) {
        if (GLOBAL_ID < num_4elems) {
            uint4* data4 = (uint4*)data;
            uint4 zeros = { 0u, 0u, 0u, 0u };
            data4[GLOBAL_ID] = zeros;
        }
    } else {
        if (GLOBAL_ID < num_4elems-1) {
            uint4* data4 = (uint4*)data;
            uint4 zeros = { 0u, 0u, 0u, 0u };
            data4[GLOBAL_ID] = zeros;
        } else {
            int global_id = GLOBAL_ID * 4;
            while (global_id < num_elements) {
                data[global_id] = 0u;
                global_id++;
            }
        }
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
    int num_4elems;

    num_4elems = (int)ceilf((float)num_elements/4.f);

    if (num_4elems <= 512) {
        blockSize = num_4elems;
        gridSize = 1;
    } else {
        blockSize = 512;
        gridSize = (int)ceilf((float)num_4elems / 512.);
    }

    kernel_steNZeros4<< <gridSize, blockSize, 0, stream >> > (data, num_elements, num_4elems);
    gpuErrchk(cudaGetLastError());
    /*else {
        kernel_setNZeros<T, (T)0> << <gridSize, blockSize, 0, stream >> > (data, num_elements);
        gpuErrchk(cudaGetLastError());
    }*/
}

template
void launch_setNZeros<uint>(uint* data, int num_elements, cudaStream_t stream);
/*template
void launch_setNZeros<char>(char* data, int num_elements, cudaStream_t stream);*/
