// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__
void uchar_convert(float *input, unsigned char *output, int size)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < size) {
    output[i] = (unsigned char) (255 * input[i]);
  }
}

__global__
void grayscale_convert(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  int idx = i * width + j;
  int max_size = width * height;
  
  // if (i == 0 && j < 3)
  //   printf("  kernel:: for idx %d (%d, %d)\n", idx, i, j);

  if (idx < max_size && i < height && j < width) {
    // channels must be 3
		unsigned char r = input[channels*idx];
		unsigned char g = input[channels*idx + 1];
		unsigned char b = input[channels*idx + 2];
		output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }

  // if (i == 0 && j < 3)    
  //   printf("kernel:: output[%d] = %u\n", idx, output[idx]);
}

__global__
void histo_kernel(unsigned char *buffer, int size, unsigned int *histo)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  while (i < size) {
    atomicAdd(&(histo[buffer[i]]), 1);
    i += stride;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  wbLog(TRACE, "Image stats: w=", imageWidth, ", h=", imageHeight, ", c=", imageChannels);
  
  // Kernel 1: uchar_convert
  wbLog(TRACE, "starting Kernel 1: uchar_convert");
  int uchar_convert_size = imageWidth * imageHeight * imageChannels;
  float *d_flt_k1_InputImageData;
  unsigned char *d_uch_k1_OutputImageData;
  unsigned char *h_uch_k1_OutputImageData = (unsigned char *) malloc(uchar_convert_size * sizeof(unsigned char));
  wbCheck(cudaMalloc((void**) &d_flt_k1_InputImageData, uchar_convert_size * sizeof(float)));
  wbCheck(cudaMalloc((void**) &d_uch_k1_OutputImageData, uchar_convert_size * sizeof(unsigned char)));
  wbCheck(cudaMemcpy(d_flt_k1_InputImageData, hostInputImageData, uchar_convert_size * sizeof(float),
                     cudaMemcpyHostToDevice));
  dim3 k1_DimBlock(256, 1, 1);
  dim3 k1_DimGrid(ceil(1.0*uchar_convert_size/k1_DimBlock.x), 1, 1);
  wbLog(TRACE, "The k1_DimGrid is ", k1_DimGrid.z, "x", k1_DimGrid.y, "x", k1_DimGrid.x);
  wbLog(TRACE, "The k1_DimBlock is ", k1_DimBlock.z, "x", k1_DimBlock.y, "x", k1_DimBlock.x);
  uchar_convert<<<k1_DimGrid, k1_DimBlock>>>(d_flt_k1_InputImageData, d_uch_k1_OutputImageData, uchar_convert_size);
  wbCheck(cudaDeviceSynchronize());
  wbCheck(cudaMemcpy(h_uch_k1_OutputImageData, d_uch_k1_OutputImageData, uchar_convert_size * sizeof(unsigned char),
                     cudaMemcpyDeviceToHost));
  wbCheck(cudaFree(d_flt_k1_InputImageData));
  wbCheck(cudaFree(d_uch_k1_OutputImageData));
  wbLog(TRACE, "finished Kernel 1: uchar_convert");

  // for (int i = 0; i < 9; i++)
  //   printf("uchar[%d] = %u\n", i, h_uch_k1_OutputImageData[i]);

  // Kernel 2: grayscale_convert
  wbLog(TRACE, "starting Kernel 2: grayscale_convert");
  int grayscale_convert_out_size = imageWidth * imageHeight;
  unsigned char *d_uch_k2_InputImageData;
  unsigned char *d_uch_k2_OutputImageData;
  unsigned char *h_uch_k2_OutputImageData = (unsigned char *) malloc(grayscale_convert_out_size * sizeof(unsigned char));
  wbCheck(cudaMalloc((void**) &d_uch_k2_InputImageData, uchar_convert_size * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void**) &d_uch_k2_OutputImageData, grayscale_convert_out_size * sizeof(unsigned char)));

  wbCheck(cudaMemcpy(d_uch_k2_InputImageData, h_uch_k1_OutputImageData, uchar_convert_size * sizeof(unsigned char),
                     cudaMemcpyHostToDevice));
  dim3 k2_DimBlock(32, 32, 1);
  dim3 k2_DimGrid(ceil(1.0*imageWidth/k2_DimBlock.x), ceil(1.0*imageHeight/k2_DimBlock.y), 1);
  wbLog(TRACE, "The k2_DimGrid is ", k2_DimGrid.z, "x", k2_DimGrid.y, "x", k2_DimGrid.x);
  wbLog(TRACE, "The k2_DimBlock is ", k2_DimBlock.z, "x", k2_DimBlock.y, "x", k2_DimBlock.x);
  grayscale_convert<<<k2_DimGrid, k2_DimBlock>>>(d_uch_k2_InputImageData, d_uch_k2_OutputImageData, imageWidth, imageHeight, imageChannels);
  wbCheck(cudaDeviceSynchronize());
  wbCheck(cudaMemcpy(h_uch_k2_OutputImageData, d_uch_k2_OutputImageData, grayscale_convert_out_size * sizeof(unsigned char),
                     cudaMemcpyDeviceToHost));
  wbCheck(cudaFree(d_uch_k2_InputImageData));
  wbCheck(cudaFree(d_uch_k2_OutputImageData));
  free(h_uch_k1_OutputImageData);
  wbLog(TRACE, "finished Kernel 2: grayscale_convert");

  // for (int i = 0; i < 9; i++)
  //   printf("grayscale[%d] = %u\n", i, h_uch_k2_OutputImageData[i]);

  // Kernel 3: histo_kernel
  wbLog(TRACE, "starting Kernel 3: histo_kernel");
  unsigned char *d_uch_k3_InputImageData;
  unsigned int *d_uint_k3_OutputHistogram;
  unsigned int *h_uint_k3_OutputHistogram = (unsigned int *) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
  wbCheck(cudaMalloc((void**) &d_uch_k3_InputImageData, grayscale_convert_out_size * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void**) &d_uint_k3_OutputHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));

  wbCheck(cudaMemcpy(d_uch_k3_InputImageData, h_uch_k2_OutputImageData, grayscale_convert_out_size * sizeof(unsigned char),
                     cudaMemcpyHostToDevice));
  dim3 k3_DimBlock(256, 1, 1);
  dim3 k3_DimGrid(ceil(1.0*grayscale_convert_out_size/k3_DimBlock.x), 1, 1);
  wbLog(TRACE, "The k3_DimGrid is ", k3_DimGrid.z, "x", k3_DimGrid.y, "x", k3_DimGrid.x);
  wbLog(TRACE, "The k3_DimBlock is ", k3_DimBlock.z, "x", k3_DimBlock.y, "x", k3_DimBlock.x);
  histo_kernel<<<k3_DimGrid, k3_DimBlock>>>(d_uch_k3_InputImageData, grayscale_convert_out_size, d_uint_k3_OutputHistogram);
  wbCheck(cudaDeviceSynchronize());
  wbCheck(cudaMemcpy(h_uint_k3_OutputHistogram, d_uint_k3_OutputHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
  wbCheck(cudaFree(d_uch_k3_InputImageData));
  wbCheck(cudaFree(d_uint_k3_OutputHistogram));
  free(h_uch_k2_OutputImageData); // needed ???
  wbLog(TRACE, "finished Kernel 3: histo_kernel");

  // for (int i = 30; i < 36; i++)
  //   wbLog(TRACE, "histo[", i, "] = ", h_uint_k3_OutputHistogram[i]);

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
