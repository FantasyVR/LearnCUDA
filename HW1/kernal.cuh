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
__global__ void HW1(uchar3* d_out, uchar3* d_in);


bool readImage(const char* filename, imageInfo* ii);
void writeImage(const char* filename, imageInfo* ii, const uchar3 *d_out);
void color2gray(imageInfo* ii, uchar3* h_out);

void exec(const char* inputFile, const char* outputFile);
#endif // 

