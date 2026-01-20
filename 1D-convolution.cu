/*
Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter),
compute the convolved output. The convolution should be performed with a "valid" boundary condition,
meaning the kernel is only applied where it fully overlaps with the input.

The input consists of two arrays:
- input:  A 1D array of 32-bit floating-point numbers.
- kernel: A 1D array of 32-bit floating-point numbers representing the convolution kernel.

The output should be written to the output array, which will have a size of:
    output_size = input_size - kernel_size + 1

The convolution operation is defined mathematically as:
    output[i] = sum_{k=0}^{kernel_size-1} input[i + k] * kernel[k]
    where i ranges from 0 to output_size-1.
*/
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
        int i= blockDim.x * blockIdx.x + threadIdx.x;
        int output_size=input_size-kernel_size+1;
        if(i<output_size){
            float sum=0.0f;
        for(int j=0;j<kernel_size;j++){
            sum+=input[i+j]*kernel[j];
        }
        output[i]=sum;
        }
        }
                                      

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
