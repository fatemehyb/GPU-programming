/*
Interleave Arrays – CUDA Implementation

Description:
This program interleaves two input arrays A and B, each containing N
32-bit floating point numbers, into a single output array of length 2N.
The output array is constructed such that elements alternate between A
and B in the following order:

    output[2*i]     = A[i]
    output[2*i + 1] = B[i]

Each CUDA thread processes one index i in the range [0, N).

Constraints:
- Uses only native CUDA features (no external libraries).
- The solve function signature remains unchanged.
- The final result is stored in the output array (device memory).
*/
#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<N){
        int out_idx=2*idx;
        output[out_idx]=A[idx];
        output[out_idx+1]=B[idx];
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
