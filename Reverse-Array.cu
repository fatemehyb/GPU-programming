#include <cuda_runtime.h>
/*
Reverse Array – In-place (CUDA)

Implement a program that reverses an array of 32-bit floating point numbers in-place.
The final result must be stored back in `input`.

No external libraries are used.
*/
__global__ void reverse_array(float* input, int N) {
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<(N/2)){
        float tmp=input[idx];

        input[idx]=input[N-1-idx];
        input[N-1-idx]=tmp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
