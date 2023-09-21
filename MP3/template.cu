
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
// thread block size for tiling
#define BLOCK_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float stA[BLOCK_WIDTH][BLOCK_WIDTH];
  __shared__ float stB[BLOCK_WIDTH][BLOCK_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = bx * blockDim.x + tx;
  int row = by * blockDim.y + ty;

  int WIDTH = numAColumns; // or numBRows

  // over q tiles, load and compute
  float result = 0;
  for (int q = 0; q < ceil((1.0 * WIDTH) / BLOCK_WIDTH); q++) {
    // Phase 1: load from source matricies into shared mem constructs
    int offsetA = q * BLOCK_WIDTH + tx; // which col
    int offsetB = q * BLOCK_WIDTH + ty; // which row
    int threadTargetIdxA = row * numAColumns + offsetA;
    int threadTargetIdxB = offsetB * numBColumns + col;

    // handle halo cells
    if (row < numCRows && (q*BLOCK_WIDTH+tx) < WIDTH)
      stA[ty][tx] = A[threadTargetIdxA];
    else {
      //printf("Kernel A halo at thread %dx%d\n", tx, ty);
      stA[ty][tx] = 0;
    }

    // handle halo cells
    if ((q*BLOCK_WIDTH+ty) < WIDTH && col < numCColumns)
      stB[ty][tx] = B[threadTargetIdxB];
    else {
      //printf("Kernel B halo at thread %dx%d\n", tx, ty);
      stB[ty][tx] = 0;
    }
    __syncthreads();
    
    // Phase 2: compute matrix mul with smaller subunit matricies
    if (col < numCColumns && row < numCRows) {
      for (int i = 0; i < BLOCK_WIDTH; i++) {
        result += stA[ty][i] * stB[i][tx];
      }
    }
    __syncthreads();
  }
  
  if (col < numCColumns && row < numCRows) {
    C[IDX_2D(col, row, numCColumns)] = result;
  }
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
  dim3 DimGrid(
    ceil((1.0*numCColumns)/BLOCK_WIDTH),
    ceil((1.0*numCRows)/BLOCK_WIDTH),
    1
  );
  dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  wbLog(TRACE, "The dimensions of DimGrid are ", DimGrid.x, " x ", DimGrid.y, " x ", DimGrid.z);
  wbLog(TRACE, "The dimensions of DimBlock are ", DimBlock.x, " x ", DimBlock.y, " x ", DimBlock.z);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, 
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
