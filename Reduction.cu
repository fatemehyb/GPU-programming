/*
Parallel Reduction (Sum) – CUDA Implementation

Description:
This program computes the sum of all elements in an array of 32-bit
floating point numbers using GPU parallel reduction.

Each CUDA block performs a partial reduction in shared memory. The
partial sums from each block are then accumulated into a single global
output value using an atomic add. The final result is stored in the
output variable (a single float in device memory).

Constraints:
- Uses only native CUDA features (no external libraries).
- The solve function signature remains unchanged.
- The final result is stored in the output variable (device memory).
*/
#include <cuda_runtime.h>
__global__ void reduce_sum_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int tid=threadIdx.x;
    int idx=blockDim.x * blockIdx.x +tid;
    float val=(idx<N)?input[idx]:0.0f;
    //shared memory is used by threads inside the the same block
    //initialize shared memory
    sdata[tid]=val;
    //wait for all threads to complte initiallizing shared memory
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // Initialize output to 0
    cudaMemset(output,0,sizeof(float));

    int threadsPerBlock=256;
    int BlockPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;//calculate ceiling

    size_t sharedBytes= threadsPerBlock * sizeof(float);
    
    reduce_sum_kernel<<<BlockPerGrid, threadsPerBlock, sharedBytes>>>(input,output, N);
    cudaDeviceSynchronize();

}

