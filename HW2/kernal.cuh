#ifndef KERNEL_H
#define KERNEL_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
typedef struct ImageInfo
{
	ImageInfo():width(0),height(0),resolution(0),image(NULL){}
	int width;
	int height;
	int resolution;
	unsigned char* image;
}imageInfo;

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}
#endif
template <typename T>
void check(T result, char const *const func, const char *const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		DEVICE_RESET
			// Make sure we call CUDA Device Reset before exiting
			exit(EXIT_FAILURE);
	}
}
/*
	将RGB图像的三个通道分开，AOS变为SOA
	RGBRGBRGB... -> RRRGGGBBB...
*/
__global__ void separateChannels(unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel,
	const uchar3* const inputImageRGB,
	int numRows, int numCols);
/*
	将三个通道合并成RGB图像， SOA变为AOS
	RRRGGGBBB... -> RGBRGBRGB...
*/
__global__
void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar3* const outputImageRGB,
	int numRows,
	int numCols);
/*
  对通道里的像素进行blur操作
  其中filter中放的是weights
*/
__global__
void gaussian_blur(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth);
void allocateMemoryAndCopyToGPU(imageInfo *ii, const float* const h_filter, const size_t filterWidth);
void cleanGPUMemory();
bool readImage(const char* filename, imageInfo* ii);
void writeImage(const char* filename, imageInfo* ii, const unsigned char *h_blurImage);
void your_gaussian_blur(imageInfo* ii, unsigned char* h_blurImage, const float *const h_filter, size_t filterWidth);

void exec(const char* inputFile, const char* outputFile);
#endif // 

