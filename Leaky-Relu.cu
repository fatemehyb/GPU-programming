/*
Leaky ReLU Activation – CUDA Implementation

Description:
This program applies the Leaky ReLU activation function to a vector of
32-bit floating point numbers on the GPU. For each element x in the input
vector, the Leaky ReLU function is defined as:

    LeakyReLU(x) = x                  if x >= 0
                   0.01 * x           if x < 0

The small positive constant (leaky coefficient) is fixed at 0.01 in this
implementation. Unlike standard ReLU, Leaky ReLU allows a small, non-zero
gradient for negative input values. Each CUDA thread processes one element
of the input vector, and the result is written to the output vector in
device memory.

Constraints:
- No external libraries are used.
- The solve function signature remains unchanged.
- The final result is stored in the output array.
*/

#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float x = input[idx];
        output[idx] = (x >= 0.0f) ? x : 0.01f * x;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
