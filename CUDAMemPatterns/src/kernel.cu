#include "./kernel.h"

#define GLOBAL_ID ((blockIdx.x * blockDim.x) + threadIdx.x)

template <typename O>
__device__ O operate(O i_data){
    return i_data;
}

template <typename I, typename O, typename I2, typename Operation, typename... operations>
__device__ O operate(I i_data, binary_operation<Operation, I, I2, O> op, operations... ops){

    if (op.parameter == scalar) {
        O temp = op.nv_operator(i_data, op.scalar);
        return operate(temp, ops...);
    } else {
        // we want to have access to I2 in order to ask for the type size for optimizing
        O temp = op.nv_operator(i_data, op.pointer[GLOBAL_ID]);
        return operate(temp, ops...);
    }
}

template<typename I, typename O, typename... operations>
__global__ void cuda_transform(I* i_data, O* o_data, operations... ops) {//binary_operation<binary_sum<float>, float> op1, binary_operation<binary_sum<float>, float> op2, binary_operation<binary_sum<float>, float> op3) {
    /*binary_operation<float> op1 = {scalar, binary_sum<float>, 1};
    binary_operation<float> op2 = {scalar, binary_sum<float>, 1};
    binary_operation<float> op3 = {scalar, binary_sum<float>, 1};*/

    /*O temp = i_data[GLOBAL_ID];

    temp = op1.nv_operator(temp,op1.scalar);
    temp = op2.nv_operator(temp,op2.scalar);
    temp = op3.nv_operator(temp,op3.scalar);*/

    //o_data[GLOBAL_ID] = temp;
    o_data[GLOBAL_ID] = operate(i_data[GLOBAL_ID], ops...);
}

void test_mult_sum_div_float(float* data, dim3 data_dims, cudaStream_t stream) {
    // We don't think about step or ROI's yet.
    dim3 thread_block(128);
    dim3 grid(1);

    binary_operation<binary_mul<float>, float> op1 = {scalar, 5.f};
    binary_operation<binary_sum<float>, float> op2 = {pointer, 0.f, data};
    binary_operation<binary_div<float>, float> op3 = {scalar, 2.f};

    cuda_transform<<<grid, thread_block, 0, stream>>>(data, data, op1, op2, op3);
    gpuErrchk(cudaGetLastError());
}
