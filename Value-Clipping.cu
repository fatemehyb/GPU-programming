/*
Clipping (Clamp) – CUDA Implementation

Description:
This program performs element-wise clipping on a 1D input vector of
32-bit floating point numbers. Given a lower bound `lo` and an upper bound
`hi`, each element x in the input tensor is clipped to the range [lo, hi]:

    clip(x) = lo        if x < lo
              hi        if x > hi
              x         otherwise

The result is written to the output tensor. Each CUDA thread processes one
element independently.

Constraints:
- Uses only native CUDA features (no external libraries).
- The solve function signature remains unchanged.
- The final result is stored in the output tensor (device memory).
*/
#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<N){
        if(input[idx]<lo){
            output[idx]=lo;
        }
        else if(input[idx]>hi){
            output[idx]=hi;
        }
        else output[idx]=input[idx];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
