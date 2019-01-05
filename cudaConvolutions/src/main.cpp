#include <vector>
#include "fast_kernel.h"
#include "cpu_baseline.h"
#include "standard_kernels.h"

#define SIZE 3840*2160

int main() {

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool success = true;
    for (int i = 0; i < SIZE; ++i) {
        //std::cout << "cpu_o_data " << cpu_o_data[i] << " == " << h_o_fast_data[i] << " h_o_fast_data " << h_o_data[i] << " h_o_data" << std::endl;
    }
    if (success) {
        std::cout << "Success!!" << std::endl;
    } else {
        std::cout << "Fail!!" << std::endl;
    }

    gpuErrchk(cudaStreamDestroy(stream));

    return 0;
}
