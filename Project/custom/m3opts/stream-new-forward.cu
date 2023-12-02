#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define NUM_STREAMS 3

__global__ void conv_forward_kernel(const int streamNum, float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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
    int b = blockIdx.z + (ceil(1.0f * B / NUM_STREAMS) * streamNum); // batch num is based on streamNum iteration
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y; // target h of output
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x; // target w of output

    // each thread ran should be within output bounds, otherwise return
    // b >= B check because splitting of grid Z may not be bounded by B
    if (w < 0 || w >= W_out || h < 0 || h >= H_out || b >= B)
        return;

    float acc = 0.0f;
    for (int c = 0; c < C; c++) { // sum over all input channels
        for (int p = 0; p < K; p++) { // loop over KxK filter
            for (int q = 0; q < K; q++) {
                int h_idx = (h * S + p);
                int w_idx = (w * S + q);
                if (!(w_idx < 0 || w_idx >= W || h_idx < 0 || h_idx >= H)) {
                    acc += in_4d(b, c, h_idx, w_idx) * mask_4d(m, c, p, q);
                }
            }
        }
    }

    // after accumulating, set to output value
    out_4d(b, m, h, w) = acc;
    // atomicAdd(&(out_4d(b, m, h, w)), acc);

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

    // device mask stream/event
    cudaStream_t dms;
    cudaStreamCreate(&dms);
    cudaEvent_t dme;
    cudaEventCreate(&dme);

    // would be async but we need CUDA 11.3+ for that. Minimal impact anyways
    wbCheck(cudaMalloc((void **)device_output_ptr, dop_sz));
    wbCheck(cudaMalloc((void **)device_input_ptr, dip_sz));
    wbCheck(cudaMalloc((void **)device_mask_ptr, dmp_sz));

    // register as pinned mem, not paged mem for GPU async transfers
    cudaHostRegister((void *) host_output, dop_sz, 0);
    cudaHostRegister((void *) host_input, dip_sz, 0);
    cudaHostRegister((void *) host_mask, dmp_sz, 0);

    // async memcpy here, record event after done to signal before kernel
    wbCheck(cudaMemcpyAsync(*device_mask_ptr, host_mask, dmp_sz, cudaMemcpyHostToDevice, dms));
    cudaEventRecord(dme, dms);

    int streamSize = ceil(1.0f * B / NUM_STREAMS); // proportion of B to batch process in each stream

    int W_size = ceil(1.0f*W_out/TILE_WIDTH); // number of horizontal tiles per output map
    int H_size = ceil(1.0f*H_out/TILE_WIDTH); // number of vertical tiles per output map
    int tileNums = H_size * W_size; // total number of tiles per map
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1); // output tile for untiled code
    dim3 DimGrid(M, tileNums, streamSize);
    std::cout<<"DimBlock: "<<DimBlock.x<<"x"<<DimBlock.y<<"x"<<DimBlock.z<<std::endl;
    std::cout<<"DimGrid: "<<DimGrid.x<<"x"<<DimGrid.y<<"x"<<DimGrid.z<<std::endl;

    cudaStream_t* streams = (cudaStream_t*) malloc(NUM_STREAMS * sizeof(cudaStream_t));

    for (int streamNum = 0; streamNum < NUM_STREAMS; streamNum++) {
        cudaStreamCreate(&(streams[streamNum]));
    }

    bool ended = false;
    for (int streamNum = 0; streamNum < NUM_STREAMS; streamNum++) {
        int offset = streamNum * streamSize; // number of Bs to skip for start of stream
        int outStreamBytes = streamSize * M * H_out * W_out * sizeof(float); // how many Bs to transfer back to host
        int inStreamBytes = streamSize * C * H * W * sizeof(float); // how many Bs to transfer to kernel

        // compute if the stream bytes on top of the offset will overflow the max bounds sizes for input and output
        int outDiff = dop_sz - ((offset * M * H_out * W_out * sizeof(float)) + outStreamBytes);
        int inDiff = dip_sz - ((offset * C * H * W * sizeof(float)) + inStreamBytes);

        // if we already ended, just skip this stream
        if (ended)
            continue;
        // if we are at or past the max input or output bounds, and we have not reached the end, mark as ended
        if((outDiff <= 0 || inDiff <= 0) && !ended)
            ended = true;

        // logging
        // std::cout<<"Starting stream: "<<streamNum<<", offset: "<< offset << ", outB: "<< outStreamBytes << ", inB: "<< inStreamBytes <<", Ended: "<<ended<<std::endl;

        // if we go past max output bounds, add the negative diff back (clamping op)
        if(outDiff < 0)
            outStreamBytes += outDiff;
        // same for input bounds
        if(inDiff < 0)
            inStreamBytes += inDiff;

        // general "stacking" idea from Mark Harris of Nvidia https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
        // changed up drastically to account for not-nice parameters and leftover work conditions, along with wait event
        cudaMemcpyAsync(&(*device_input_ptr)[offset * C * H * W], &host_input[offset * C * H * W], inStreamBytes, cudaMemcpyHostToDevice, streams[streamNum]);
        cudaStreamWaitEvent(streams[streamNum], dme, 0);
        conv_forward_kernel<<<DimGrid, DimBlock, 0, streams[streamNum]>>>(streamNum, *device_output_ptr, *device_input_ptr, *device_mask_ptr, B, M, C, H, W, K, S);
        cudaMemcpyAsync((void *) &host_output[offset * M * H_out * W_out], &(*device_output_ptr)[offset * M * H_out * W_out], outStreamBytes, cudaMemcpyDeviceToHost, streams[streamNum]);
    }

    for (int streamNum = 0; streamNum < NUM_STREAMS; streamNum++) {
        cudaStreamDestroy(streams[streamNum]);
    }

    cudaEventDestroy(dme);
    cudaStreamDestroy(dms);

    cudaHostUnregister((void *) host_output);
    cudaHostUnregister((void *) host_input);
    cudaHostUnregister((void *) host_mask);

    // not needed because we sync after anyways
    // cudaDeviceSynchronize();

    free(streams);

    #undef wbCheck
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel

    // due to limitations of this fn definition, the logic for running the kernels is all in the prolog fn
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

    // All we need to do here is cleanup

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
