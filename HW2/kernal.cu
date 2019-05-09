#include "kernal.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include <stb_image_write.h>
#include <iostream>
uchar3 *d_rgbImage, *d_blurImage;
unsigned char *d_red, *d_green, *d_blue;
unsigned char *d_blurRed, *d_blurGreen, *d_blurBlue;
float *d_filter;
void your_gaussian_blur(imageInfo* ii, unsigned char* h_blurImage, const float *const h_filter, size_t filterWidth)
{
	// Allocate GPU memories
	allocateMemoryAndCopyToGPU(ii, h_filter, filterWidth);
	
	// Step 1: 将RGB三通道分开
	separateChannels<<<ii->height,ii->width>>>(d_red, d_green, d_blue, d_rgbImage, ii->height, ii->width);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	// Step 2: 对每个通过分别进行Blur操作
	gaussian_blur << <ii->height, ii->width >> > (d_red, d_blurRed,ii->height,ii->width,d_filter,filterWidth);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	gaussian_blur << <ii->height, ii->width >> > (d_green, d_blurGreen, ii->height, ii->width, d_filter, filterWidth);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	gaussian_blur << <ii->height, ii->width >> > (d_blue, d_blurBlue, ii->height, ii->width, d_filter, filterWidth);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	// Step 3: 将三通道合并
	recombineChannels << <ii->height, ii->width >> > (d_blurRed, d_blurGreen, d_blurBlue ,d_blurImage,ii->height, ii->width);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	// Step 4: 将GPU上blur的Image 传到 CPU内存上
	checkCudaErrors(cudaMemcpy(h_blurImage, d_blurImage, ii->resolution * sizeof(uchar3), cudaMemcpyDeviceToHost));
	
	// 释放显存空间
	cleanGPUMemory();
}
void allocateMemoryAndCopyToGPU(imageInfo *ii, const float* const h_filter, const size_t filterWidth)
{
	int numPixels = ii->resolution;
	// allocate memory on GPU for picture
	checkCudaErrors(cudaMalloc((void**)&d_rgbImage, numPixels * sizeof(uchar3)));
	checkCudaErrors(cudaMalloc((void**)&d_blurImage, numPixels * sizeof(uchar3)));
	// Copy Image from CPU to GPU
	checkCudaErrors(cudaMemcpy(d_rgbImage, ii->image, numPixels * sizeof(uchar3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_blurImage, 0, numPixels * sizeof(uchar3)));
	// allocate channels for image
	checkCudaErrors(cudaMalloc((void**)&d_red, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void**)&d_green, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void**)&d_blue, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(d_red, 0, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(d_green, 0, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(d_blue, 0, numPixels * sizeof(unsigned char)));
	// allocate channels for blured image
	checkCudaErrors(cudaMalloc((void**)&d_blurRed, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void**)&d_blurGreen, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc((void**)&d_blurBlue, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(d_blurRed, 0, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(d_blurGreen, 0, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMemset(d_blurBlue, 0, numPixels * sizeof(unsigned char)));

	// Allocate memory for filter
	checkCudaErrors(cudaMalloc((void**)&d_filter, filterWidth * filterWidth * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
}
void cleanGPUMemory()
{
	// Free GPU memory
	checkCudaErrors(cudaFree(d_rgbImage));
	checkCudaErrors(cudaFree(d_blurImage));
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_blurRed));
	checkCudaErrors(cudaFree(d_blurGreen));
	checkCudaErrors(cudaFree(d_blurBlue));
	checkCudaErrors(cudaFree(d_filter));
}
__global__ void separateChannels(unsigned char * const redChannel, unsigned char * const greenChannel, 
	unsigned char * const blueChannel, const uchar3 * const inputImageRGB, int numRows, int numCols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	redChannel[idx] = inputImageRGB[idx].x;
	greenChannel[idx] = inputImageRGB[idx].y;
	blueChannel[idx] = inputImageRGB[idx].z;
}

__global__ void recombineChannels(const unsigned char * const redChannel, const unsigned char * const greenChannel, 
	const unsigned char * const blueChannel, uchar3 * const outputImageRGB, int numRows, int numCols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char red = redChannel[idx];
	unsigned char green = greenChannel[idx];
	unsigned char blue = blueChannel[idx];
	outputImageRGB[idx] = make_uchar3(red, green, blue);
}

__global__ void gaussian_blur(const unsigned char * const inputChannel, unsigned char * const outputChannel, 
	int numRows, int numCols, const float * const filter, const int filterWidth)
{
	// compute current row and cloumn index
	int row = blockIdx.x;
	int col = threadIdx.x;
	int currentTidx = row * blockDim.x + col;
	// find its neighbors' index 
	int left  = (col - filterWidth / 2) ; left = left < 0 ? 0 : left;
	int right = (col + filterWidth / 2) ; right = right < numCols? right : numCols;
	int up    = (row - filterWidth / 2) ; up = up < 0 ? 0 : up;
	int below = (row + filterWidth / 2) ; below = below < numRows ? below : numRows;

	for (size_t i = left; i < left + filterWidth && i <= right; i++)
		for (size_t j = up; j < up + filterWidth && j <= below; j++)
		{
			int tIdx = j * blockDim.x + i;
			int x = i - col;
			int y = j - row;
			int filterIdx = (y - 1) * filterWidth + x;
			filterIdx = 0 - filterIdx;
			outputChannel[currentTidx] += filter[filterIdx] * inputChannel[tIdx];
		} 
}

bool readImage(const char * filename, imageInfo* ii)
{
	int width, height, channels_in_file;
	ii->image = stbi_load(filename, &width, &height, &channels_in_file, 0);
	if (ii->image == NULL)
	{
		std::cerr << "Failed to load Image at: " << filename << std::endl;
		return false;
	}
	ii->height = height;
	ii->width = width;
	ii->resolution = height * width;
	return true;
}

void writeImage(const char* filename, imageInfo* ii, const unsigned char *h_blurImage)
{
	int res = stbi_write_jpg(filename, ii->width, ii->height, 3, h_blurImage, 0);
	if (res == 0)
	{
		std::cout << "Failed to write image file" << std::endl;
		return;
	}
	std::cout << "Write Image Successfully to: " << filename << std::endl;
}

void exec(const char * inputFile, const char * outputFile)
{
	// 读取图片
	imageInfo* ii = new imageInfo();
	bool res = readImage(inputFile, ii);
	if (!res) return;
	unsigned char *h_out = (unsigned char*)malloc(sizeof(uchar3) * ii->resolution);
	if (h_out == NULL)
	{
		std::cout << "Failed to malloc h_out space" << std::endl;
		return;
	}
	float h_filter[] = { 0.0,0.2,0.0,
						0.2,0.2,0.2,
						0.0,0.2,0.0 };
	// Blur 图片
	your_gaussian_blur(ii,h_out,h_filter,3);
	// 保存 Blur 图片
	writeImage(outputFile, ii, h_out);
	// 释放空间
	free(ii);
	free(h_out);
	h_out = NULL;
}