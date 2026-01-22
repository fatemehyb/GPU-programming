/*
Count Array Element – CUDA Implementation

Description:
This program counts how many elements in an array of 32-bit integers are
equal to a given value K. The input array has length N and resides in GPU
memory. Each CUDA thread checks one element of the array. If the element
equals K, the thread increments a shared counter in global memory using
an atomic operation.

The final count is stored in the output variable (a single integer in
device memory).

Constraints:
- No external libraries are used.
- The solve function signature remains unchanged.
- The final result is stored in the output variable.
*/

#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int idx= blockDim.x * blockIdx.x +threadIdx.x;
    if(idx<N && input[idx]==K){
        atomicAdd(output,1);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}
