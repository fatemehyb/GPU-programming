/*
Rainbow Table (Parallel Iterative Hashing) – CUDA Implementation

Description:
This program performs R rounds of parallel hashing on an array of 32-bit
integers using the provided FNV-1a hash function. Each element is hashed
iteratively R times, where the output of one round becomes the input to
the next round:

    x0 = input[i]
    x1 = fnv1a_hash(x0)
    x2 = fnv1a_hash(x1)
    ...
    xR = fnv1a_hash(xR-1)

The final value xR is written to output[i]. Each CUDA thread processes one
array element independently, enabling parallel computation across the GPU.

Constraints:
- No external libraries are used.
- The solve function signature remains unchanged.
- The final result is stored in the output array (device memory).
*/
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int idx = blockDim.x * blockIdx.x +threadIdx.x;
    if(idx<N){
        int tmp=input[idx];
        for(int i=0;i<R;i++){
            tmp=fnv1a_hash(tmp);
        }
        output[idx]=tmp;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
