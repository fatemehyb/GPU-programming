#include <cuda_runtime.h>
/*
ReLU (Rectified Linear Unit) Activation – CUDA Implementation

Description:
This program applies the ReLU activation function to a vector of 32-bit
floating point numbers on the GPU. For each element x in the input vector,
the ReLU function is defined as:

    ReLU(x) = max(0.0f, x)

All negative values are set to zero, while positive values remain unchanged.
Each CUDA thread processes one element of the input vector. The result is
written to the output vector in device memory.

Constraints:
- No external libraries are used.
- The solve function signature remains unchanged.
- The final result is stored in the output array.
*/
__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx=blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<N){
        if(input[idx]<0)
        output[idx]=0;
        else
        output[idx]=input[idx];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
