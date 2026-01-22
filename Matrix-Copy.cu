/*
Matrix Copy – CUDA Implementation

Description:
This program copies an N × N matrix of 32-bit floating point numbers from
the input array A to the output array B on the GPU. The matrix is stored
in row-major order as a contiguous 1D array of length N*N. Each CUDA
thread copies exactly one element, so that for every valid index i:

    B[i] = A[i]

This is a direct element-wise copy with no modification of the data.

Constraints:
- No external libraries are used.
- The solve function signature remains unchanged.
- The final result is stored in the output matrix B (device memory).
*/
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx<N * N){
        B[idx]=A[idx];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
