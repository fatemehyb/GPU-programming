/*
Count 2D Array Element (Easy) – CUDA Implementation

Description:
Counts how many elements in an N × M 2D array of 32-bit integers are equal
to a given value K. The 2D array is stored in row-major order as a flat
1D array:

    idx = row * M + col

Each CUDA thread checks one element (row, col). If it equals K, the thread
increments the global counter using atomicAdd. The final count is stored
in the output variable (device memory).

Constraints:
- No external libraries are used.
- The solve function signature remains unchanged.
- The final result is stored in the output variable.
*/
#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if(idx<M && idy<N){
        if(input[idy*M+idx]==K){
            atomicAdd(output,1);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}
