/* Write a program to invert the colors of an image. The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each component is an 8-bit unsigned integer (unsigned char).

Color inversion is performed by subtracting each color component (R, G, B) from 255. The Alpha component should remain unchanged.

The input array image will contain width * height * 4 elements. The first 4 elements represent the RGBA values of the top-left pixel, the next 4 elements represent the pixel to its right, and so on. */
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int pixelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int total_pixels = width * height;
    if(pixelIndex<total_pixels){
        int base=pixelIndex * 4;
        image[base+0]=255-image[base+0];
        image[base+1]=255-image[base+1];
        image[base+2]=255-image[base+2];
        
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
