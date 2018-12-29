#include <vector>
#include "./kernel.h"

#define SIZE 128

template <typename O>
O operate(int i, O i_data){
    return i_data;
}

template <typename I, typename O, typename I2, typename... operations>
O operate(int i, I i_data, cpu_binary_operation<I, I2, O> op, operations... ops){

    if (op.parameter == scalar) {
        O temp = op.nv_operator(i_data, op.scalar);
        return operate(i, temp, ops...);
    } else {
        // we want to have access to I2 in order to ask for the type size for optimizing
        O temp = op.nv_operator(i_data, op.pointer[i]);
        return operate(i, temp, ops...);
    }

}

template<typename I, typename O, typename... operations>
void cpu_cuda_transform(I* i_data, O* o_data, operations... ops) {
    for (int i=0; i<SIZE; ++i) {
        o_data[i] = operate(i, i_data[i], ops...);
    }
}

int main() {

    float* data = (float*)malloc(sizeof(float)*SIZE);
    float* h_data = (float*)malloc(sizeof(float)*SIZE);

    for (int i=0; i<SIZE; ++i) {
        data[i] = (float)i;
    }

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    float* d_data;
    gpuErrchk(cudaMalloc(&d_data, sizeof(float)*SIZE));
    gpuErrchk(cudaMemcpyAsync(d_data, data, sizeof(float)*SIZE, cudaMemcpyHostToDevice, stream));

    dim3 size(128);
    test_mult_sum_div_float(d_data, size, stream);

    gpuErrchk(cudaMemcpyAsync(h_data, d_data, sizeof(float)*SIZE, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    cpu_binary_operation<float> op1 = {scalar, cpu_binary_mul<float>, 5};
    cpu_binary_operation<float> op2 = {pointer, cpu_binary_sum<float>, 0, data};
    cpu_binary_operation<float> op3 = {scalar, cpu_binary_div<float>, 2};

    cpu_cuda_transform(data, data, op1, op2, op3);

    std::cout << "Executed!!" << std::endl;
    bool success = true;
    for (int i=0; i<SIZE; ++i) {
        if (success) success &= 0.001 > data[i] - h_data[i];
        std::cout << "data " << data[i] << " == " << h_data[i] << " h_data" << std::endl;
    }
    free(data);

    if (success) {
        std::cout << "Success!!" << std::endl;
    } else {
        std::cout << "Fail!!" << std::endl;
    }

    return 0;
}
