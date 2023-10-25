// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void finish_scan(float *input, float *output, int len, float* blockSumInput, int blockSumInputLen) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int i = 2 * BLOCK_SIZE * bx + tx;

  // apply block sum to elements shifted over one block in block sum arr
  int blockSum = 0;
  if (bx-1 >= 0)
    blockSum = blockSumInput[bx-1];
  if (i < len)
    output[i] = input[i] + blockSum;
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];

  // load into shared memory
  int i = 2 * BLOCK_SIZE * blockIdx.x + threadIdx.x;
  if (i < len) 
    T[threadIdx.x] = input[i];
  else
    T[threadIdx.x] = 0;
  
  // printf("tidx: %d = %f\n", threadIdx.x, T[threadIdx.x]);
  // T[threadIdx.x] = i < len ? input[i] : 0.0f;

  // scan step
  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];
    // todo: need the syncthreads here?
    stride = stride*2;
  }
  __syncthreads();

  // post scan step
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE)
      T[index+stride] += T[index];
    stride = stride / 2;
  }
  __syncthreads();

  // write shared memory back to output
  if (i < len) 
    output[i] = T[threadIdx.x];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(2*BLOCK_SIZE, 1, 1);
  int numOfBlockSums = ceil(numElements * (1.0f) / DimBlock.x);
  dim3 DimGrid(numOfBlockSums, 1, 1);
  wbLog(TRACE, "The number of block sums is ", DimGrid.x, ", block size is ", DimBlock.x);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  // initial scan per block
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  // for (int i = 0; i < 12; i++) {
  //   wbLog(TRACE, "hI[", i, "] = ", hostInput[i], "; ", "hO[", i, "] = ", hostOutput[i]);
  // }

  // setup block sum scan
  float *h_blockSumInput = (float *)malloc(numOfBlockSums * sizeof(float));
  float *h_blockSumOutput = (float *)malloc(numOfBlockSums * sizeof(float));

  // pick out block sum elements into host array
  for (int i = 0; i < numOfBlockSums; i++) {
    int targetIdx = (i+1) * DimBlock.x - 1;
    if (targetIdx > numElements-1) 
      targetIdx = numElements-1;
    h_blockSumInput[i] = hostOutput[targetIdx];
    // wbLog(TRACE, "BSI[", i, "] = ", h_blockSumInput[i], " //", targetIdx);
  }

  float *d_blockSumInput;
  float *d_blockSumOutput;
  wbCheck(cudaMalloc((void **)&d_blockSumInput, numOfBlockSums * sizeof(float)));
  wbCheck(cudaMalloc((void **)&d_blockSumOutput, numOfBlockSums * sizeof(float)));
  wbCheck(cudaMemcpy(d_blockSumInput, h_blockSumInput, numOfBlockSums * sizeof(float),
                     cudaMemcpyHostToDevice));

  // scan over block sums
  dim3 BSDimGrid(1, 1, 1);
  scan<<<BSDimGrid, DimBlock>>>(d_blockSumInput, d_blockSumOutput, numOfBlockSums);
  cudaDeviceSynchronize();
  // copy, free device memory
  wbCheck(cudaMemcpy(h_blockSumOutput, d_blockSumOutput, numOfBlockSums * sizeof(float),
                     cudaMemcpyDeviceToHost));
  cudaFree(d_blockSumInput);
  cudaFree(d_blockSumOutput);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // for (int i = 0; i < numOfBlockSums; i++) {
  //   wbLog(TRACE, "BSO[", i, "] = ", h_blockSumOutput[i]);
  // }

  // post scan procedure
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemcpy(deviceInput, hostOutput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMalloc((void **)&d_blockSumInput, numOfBlockSums * sizeof(float)));
  wbCheck(cudaMemcpy(d_blockSumInput, h_blockSumOutput, numOfBlockSums * sizeof(float),
                    cudaMemcpyHostToDevice));
  finish_scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements, d_blockSumInput, numOfBlockSums);
  cudaDeviceSynchronize();

  // free referenced block sum memory
  cudaFree(d_blockSumInput);
  free(h_blockSumInput);
  free(h_blockSumOutput);

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
