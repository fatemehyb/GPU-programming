/*
Sigmoid Linear Unit (SiLU) – CUDA Implementation

Description:
This program computes the forward pass of the SiLU (Sigmoid Linear Unit)
activation function for a 1D input vector of 32-bit floating point numbers.

The SiLU function is defined element-wise as:

    SiLU(x) = x * sigmoid(x)
            = x / (1 + exp(-x))

Each CUDA thread processes one element of the input vector. The computed
SiLU value is written directly to the corresponding position in the output
vector.

Constraints:
- Uses only native CUDA features (no external libraries).
- The solve function signature remains unchanged.
- The final result is stored in the output tensor (device memory).
*/
#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<N){
        float phi=1/(1.0f+expf(-input[idx]));
        output[idx]=input[idx]*phi;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
