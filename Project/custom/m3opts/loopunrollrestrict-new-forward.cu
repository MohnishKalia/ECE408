#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    // same as grid setup
    int W_size = ceil(1.0f*W_out/TILE_WIDTH); // number of horizontal tiles per output map
    int H_size = ceil(1.0f*H_out/TILE_WIDTH); // number of vertical tiles per output map
    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y; // target h of output
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x; // target w of output

    // each thread ran should be within output bounds, otherwise return
    if (w < 0 || w >= W_out || h < 0 || h >= H_out)
        return;

    float acc = 0.0f;

    // loop unroll logic based on mask size
    if (K <= 3) {
        // low K, just do normally, no real performance gain from unroll
        for (int c = 0; c < C; c++) { // sum over all input channels
            for (int p = 0; p < K; p++) { // loop over KxK filter
                for (int q = 0; q < K; q++) {
                    int h_idx = (h * S + p);
                    int w_idx = (w * S + q);
                    // if target idx is not within input bounds, use 0, otherwise grab value
                    if (!(w_idx < 0 || w_idx >= W || h_idx < 0 || h_idx >= H))
                        acc += in_4d(b, c, h_idx, w_idx) * mask_4d(m, c, p, q);
                }
            }
        }
    } 
    else if (K <= 7)
    {
        #define Q_UNROLL 4
        // middling K value, get a few unroll ops
        for (int c = 0; c < C; c++) { // sum over all input channels
            for (int p = 0; p < K; p++) { // loop over KxK filter
                // turn q into # of loop iters to target instead of just q
                int qbnd = ceil(1.0f * K / Q_UNROLL);
                for (int q = 0; q < qbnd; q++) {
                    // calc q0..3 and w0..3
                    int h_idx = (h * S + p);
                    int q0 = q * Q_UNROLL + 0;
                    int q1 = q * Q_UNROLL + 1;
                    int q2 = q * Q_UNROLL + 2;
                    int q3 = q * Q_UNROLL + 3;
                    int w0_idx = (w * S + q0);
                    int w1_idx = (w * S + q1);
                    int w2_idx = (w * S + q2);
                    int w3_idx = (w * S + q3);
                    // if target idx is not within input bounds, or q# is too far, use 0, otherwise grab value
                    if (!(w0_idx < 0 || w0_idx >= W || h_idx < 0 || h_idx >= H) && q0 < K)
                        acc += in_4d(b, c, h_idx, w0_idx) * mask_4d(m, c, p, q0);
                    if (!(w1_idx < 0 || w1_idx >= W || h_idx < 0 || h_idx >= H) && q1 < K)
                        acc += in_4d(b, c, h_idx, w1_idx) * mask_4d(m, c, p, q1);
                    if (!(w2_idx < 0 || w2_idx >= W || h_idx < 0 || h_idx >= H) && q2 < K)
                        acc += in_4d(b, c, h_idx, w2_idx) * mask_4d(m, c, p, q2);
                    if (!(w3_idx < 0 || w3_idx >= W || h_idx < 0 || h_idx >= H) && q3 < K)
                        acc += in_4d(b, c, h_idx, w3_idx) * mask_4d(m, c, p, q3);
                }
            }
        }
        #undef Q_UNROLL
    }
    else
    {
        #define Q_UNROLL 8
        // beyond this K >= 8, performance gains fetter out
        // only a few unrolls
        for (int c = 0; c < C; c++) { // sum over all input channels
            for (int p = 0; p < K; p++) { // loop over KxK filter
                // turn q into # of loop iters to target instead of just q
                int qbnd = ceil(1.0f * K / Q_UNROLL);
                for (int q = 0; q < qbnd; q++) {
                    // calc q0..7 and w0..7
                    int h_idx = (h * S + p);
                    int q0 = q * Q_UNROLL + 0;
                    int q1 = q * Q_UNROLL + 1;
                    int q2 = q * Q_UNROLL + 2;
                    int q3 = q * Q_UNROLL + 3;
                    int q4 = q * Q_UNROLL + 4;
                    int q5 = q * Q_UNROLL + 5;
                    int q6 = q * Q_UNROLL + 6;
                    int q7 = q * Q_UNROLL + 7;
                    int w0_idx = (w * S + q0);
                    int w1_idx = (w * S + q1);
                    int w2_idx = (w * S + q2);
                    int w3_idx = (w * S + q3);
                    int w4_idx = (w * S + q4);
                    int w5_idx = (w * S + q5);
                    int w6_idx = (w * S + q6);
                    int w7_idx = (w * S + q7);
                    // if target idx is not within input bounds, or q# is too far, use 0, otherwise grab value
                    if (!(w0_idx < 0 || w0_idx >= W || h_idx < 0 || h_idx >= H) && q0 < K)
                        acc += in_4d(b, c, h_idx, w0_idx) * mask_4d(m, c, p, q0);
                    if (!(w1_idx < 0 || w1_idx >= W || h_idx < 0 || h_idx >= H) && q1 < K)
                        acc += in_4d(b, c, h_idx, w1_idx) * mask_4d(m, c, p, q1);
                    if (!(w2_idx < 0 || w2_idx >= W || h_idx < 0 || h_idx >= H) && q2 < K)
                        acc += in_4d(b, c, h_idx, w2_idx) * mask_4d(m, c, p, q2);
                    if (!(w3_idx < 0 || w3_idx >= W || h_idx < 0 || h_idx >= H) && q3 < K)
                        acc += in_4d(b, c, h_idx, w3_idx) * mask_4d(m, c, p, q3);
                    if (!(w4_idx < 0 || w4_idx >= W || h_idx < 0 || h_idx >= H) && q4 < K)
                        acc += in_4d(b, c, h_idx, w4_idx) * mask_4d(m, c, p, q4);
                    if (!(w5_idx < 0 || w5_idx >= W || h_idx < 0 || h_idx >= H) && q5 < K)
                        acc += in_4d(b, c, h_idx, w5_idx) * mask_4d(m, c, p, q5);
                    if (!(w6_idx < 0 || w6_idx >= W || h_idx < 0 || h_idx >= H) && q6 < K)
                        acc += in_4d(b, c, h_idx, w6_idx) * mask_4d(m, c, p, q6);
                    if (!(w7_idx < 0 || w7_idx >= W || h_idx < 0 || h_idx >= H) && q7 < K)
                        acc += in_4d(b, c, h_idx, w7_idx) * mask_4d(m, c, p, q7);
                }
            }
        }
        #undef Q_UNROLL
    }
    

    // after accumulating, set to output value
    out_4d(b, m, h, w) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    #define wbCheck(stmt)                                                     \
    do {                                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            std::cout<<"Failed to run stmt: "<<#stmt<<std::endl;    \
            std::cout<<"CUDA error: "<<cudaGetErrorString(err)<<std::endl;    \
            exit(-1);                                                         \
        }                                                                     \
    } while (0)

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    size_t dop_sz = B * M * H_out * W_out * sizeof(float);
    size_t dip_sz = B * C * H * W * sizeof(float);
    size_t dmp_sz = M * C * K * K * sizeof(float);
    wbCheck(cudaMalloc((void **)device_output_ptr, dop_sz));
    wbCheck(cudaMalloc((void **)device_input_ptr, dip_sz));
    wbCheck(cudaMalloc((void **)device_mask_ptr, dmp_sz));

    // wbCheck(cudaMemcpy(*device_output_ptr, host_output, dop_sz, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(*device_input_ptr, host_input, dip_sz, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(*device_mask_ptr, host_mask, dmp_sz, cudaMemcpyHostToDevice));

    #undef wbCheck
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel

    // same as inside kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int W_size = ceil(1.0f*W_out/TILE_WIDTH); // number of horizontal tiles per output map
    int H_size = ceil(1.0f*H_out/TILE_WIDTH); // number of vertical tiles per output map
    int tileNums = H_size * W_size; // total number of tiles per map
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1); // output tile for untiled code
    dim3 DimGrid(M, tileNums, B);
    std::cout<<"DimBlock: "<<DimBlock.x<<"x"<<DimBlock.y<<"x"<<DimBlock.z<<std::endl;
    std::cout<<"DimGrid: "<<DimGrid.x<<"x"<<DimGrid.y<<"x"<<DimGrid.z<<std::endl;
    conv_forward_kernel<<<DimGrid, DimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{

    #define wbCheck(stmt)                                                     \
    do {                                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            std::cout<<"Failed to run stmt: "<<#stmt<<std::endl;    \
            std::cout<<"CUDA error: "<<cudaGetErrorString(err)<<std::endl;    \
            exit(-1);                                                         \
        }                                                                     \
    } while (0)

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    size_t dop_sz = B * M * H_out * W_out * sizeof(float);

    // Copy the output back to host
    wbCheck(cudaMemcpy(host_output, device_output, dop_sz, cudaMemcpyDeviceToHost));

   
    // Free device memory
    wbCheck(cudaFree(device_input));
    wbCheck(cudaFree(device_output));
    wbCheck(cudaFree(device_mask));

    #undef wbCheck
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
