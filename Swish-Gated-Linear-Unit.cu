/*
Swish-Gated Linear Unit (SWiGLU) – CUDA Implementation

Description:
This program computes the forward pass of the Swish-Gated Linear Unit
(SWiGLU) activation function for a 1D input vector of 32-bit floating point
numbers.

Given an input tensor of shape [N], it is split into two equal halves:
- First half:  a = input[0 .. halfN-1]
- Second half: b = input[halfN .. 2*halfN-1]

SWiGLU is defined element-wise as:

    SiLU(a) = a / (1 + exp(-a))
    SWiGLU(a, b) = SiLU(a) * b

The output tensor has shape [halfN], where each element is the SWiGLU
result for the corresponding pair (a[i], b[i]).

Each CUDA thread processes one element index i in [0, halfN).

Constraints:
- Uses only native CUDA features (no external libraries).
- The solve function signature remains unchanged.
- The final result is stored in the output tensor (device memory).
*/
#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    
    if(idx<halfN){

        float a=input[idx];
        float b=input[idx+halfN];
        a=a*(1.0f/(1.0f+expf(-a)));
        output[idx]=a*b;

    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
