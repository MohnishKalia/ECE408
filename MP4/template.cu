#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
// helper for computing malloc size
#define DATA_SIZE(inputLen) (inputLen * sizeof(float))
// consts
#define TILE_WIDTH (8)
#define MASK_WIDTH (3)
#define MASK_RADIUS (MASK_WIDTH / 2)
#define BLOCK_WIDTH (TILE_WIDTH + (MASK_WIDTH - 1))

__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

//@@ Define constant memory for device kernel here

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbCheck(cudaMalloc((void **)&deviceInput, DATA_SIZE(inputLength - 3)));
  wbCheck(cudaMalloc((void **)&deviceOutput, DATA_SIZE(inputLength)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, (hostInput + 3), DATA_SIZE(inputLength - 3), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpyToSymbol(Mc, hostKernel, DATA_SIZE(kernelLength)));
  // // kernel check
  // for (size_t i = 0; i < kernelLength; i++)
  //   wbLog(TRACE, "", *(hostKernel + i));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(
    ceil(x_size/(1.0*TILE_WIDTH)),
    ceil(y_size/(1.0*TILE_WIDTH)),
    ceil(z_size/(1.0*TILE_WIDTH))
  );
  dim3 DimBlock(
    BLOCK_WIDTH,
    BLOCK_WIDTH,
    BLOCK_WIDTH
  );
  wbLog(TRACE, "The DimGrid is ", DimGrid.x, "x", DimGrid.y, "x", DimGrid.z);
  wbLog(TRACE, "The DimBlock is ", DimBlock.x, "x", DimBlock.y, "x", DimBlock.z);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy((hostOutput + 3), deviceOutput, DATA_SIZE(inputLength - 3), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
