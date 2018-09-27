#include <vector>
#include "./kernel.h"

template <class T>
void do_experiment(int num_elements, cudaStream_t stream) {
    T *gpukData, *gpumData;
    gpuErrchk(cudaMalloc(&gpukData, sizeof(T)*num_elements));
    gpuErrchk(cudaMalloc(&gpumData, sizeof(T)*num_elements));

    launch_setNZeros(gpukData, num_elements, stream);
    gpuErrchk(cudaMemsetAsync(gpumData, 0, sizeof(T) * num_elements, stream));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaFree(gpukData));
    gpuErrchk(cudaFree(gpumData));
}

int main() {
    uint *data;
    gpuErrchk(cudaMalloc(&data, sizeof(uint)));

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    launch_setAZero(data, stream);
    gpuErrchk(cudaMemsetAsync(data, 0, sizeof(uint), stream));

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaFree(data));
    // Second batch of tests

    std::vector<int> nums_elements = { 1, 32, 133, 512, 141435, 4096*64 };

    for (auto i : nums_elements) {
        do_experiment<uint>(i, stream);
        // do_experiment<char>(i, stream);
    }

    std::cout << "Executed!!" << std::endl;

    return 0;
}
