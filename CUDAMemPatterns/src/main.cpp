#include <vector>
#include "./kernel.h"

#define SIZE 128

template <typename O>
O operate(int i, O i_data){
    return i_data;
}

template <typename I, typename O, typename I2, typename... operations>
O operate(int i, I i_data, binary_operation<I, I2, O> op, operations... ops){

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
void cuda_transform(I* i_data, O* o_data, operations... ops) {
    for (int i=0; i<SIZE; ++i) {
        o_data[i] = operate(i, i_data[i], ops...);
    }
}

int main() {
    // We prepare room on the GPU for a data vector. Let's make it somewhat big.
    // For the sake of working with 2D data, let's suppose we work with a 1024*512 matrix,
    // made of float values.
    /*float *data;
    gpuErrchk(cudaMalloc(&data, sizeof(float)*1024*512));

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    dim3 data_dims(1024, 512);
    test_mult_sum_div_float(data, 2, data, 2, data_dims, stream);

    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaFree(data));*/

    float* data = (float*)malloc(sizeof(float)*SIZE);

    for (int i=0; i<SIZE; ++i) {
        data[i] = (float)i;
    }

    binary_operation<float, float, float> op1 = {};
    op1.parameter = scalar;
    op1.nv_operator = binary_sum<float, float, float>;
    op1.scalar = 1;
    binary_operation<float, float, float> op2 = {};
    op2.parameter = scalar;
    op2.nv_operator = binary_sum<float, float, float>;
    op2.scalar = 1;
    binary_operation<float, float, float> op3 = {};
    op3.parameter = scalar;
    op3.nv_operator = binary_sum<float, float, float>;
    op3.scalar = 1;

    cuda_transform(data, data, op1, op2, op3);

    std::cout << "Executed!!" << std::endl;
    for (int i=0; i<SIZE; ++i) {
        std::cout << data[i] << std::endl;
    }
    free(data);

    return 0;
}
