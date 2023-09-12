
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// helper for computing malloc size
#define DATA_SIZE(inputLen) (inputLen * sizeof(float))
// map 2d to 1d array
#define IDX_2D(x, y, stride) (y * stride + x)
// leftover from subarray chunks
#define IDX_2_ROW(i, stride) (i % stride)
// go back to nearest subarray, then divide elements by length of subarray
#define IDX_2_COL(i, stride) ((i - (i % stride)) / stride)

// Compute C = A * B
__global__ 
void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // boundary check
  if (i >= numCRows * numCColumns) return;

  int aRow = IDX_2_ROW(i, numCColumns);
  int bCol = IDX_2_COL(i, numCColumns);

  float result = 0;
  for (size_t i = 0; i < numCColumns; i++) {
    result += A[IDX_2D(i, aRow, numAColumns)] * B[IDX_2D(bCol, i, numBColumns)];
  }
  C[i] = result;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  if (numAColumns != numBRows) {
    wbLog(ERROR, "Invalid A and B dimensions for matrix multiplication.");
    return 1;
  }
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  int cInputSize = numCRows * numCColumns;
  hostC = (float *)malloc(DATA_SIZE(cInputSize));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int aInputSize = numARows * numAColumns;
  int bInputSize = numBRows * numBColumns;
  wbCheck(cudaMalloc((void **)&deviceA, DATA_SIZE(aInputSize)));
  wbCheck(cudaMalloc((void **)&deviceB, DATA_SIZE(bInputSize)));
  wbCheck(cudaMalloc((void **)&deviceC, DATA_SIZE(cInputSize)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, DATA_SIZE(aInputSize), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, DATA_SIZE(bInputSize), cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(cInputSize/256, 1, 1);
  // if leftovers, add one more
  if ((cInputSize % 256) != 0)
    DimGrid.x++;
  dim3 DimBlock(256, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, 
                                        numARows, numAColumns, 
                                        numBRows, numBColumns,
                                        numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, DATA_SIZE(cInputSize), cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
