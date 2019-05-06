#include "kernal.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include <stb_image_write.h>
#include <iostream>
__global__ void HW1(unsigned char * d_out, unsigned char * d_in)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	printf("trheadIdx is: %d \n", threadId);
	d_out[threadId] = d_in[threadId];
}

void color2gray(imageInfo* ii,unsigned char * h_out)
{
	unsigned char * d_in, unsigned char * d_out;

	int numPixels = ii->resolution;
	// allocate memory on GPU for picture
	checkCudaErrors(cudaMalloc((void**)&d_in, numPixels * sizeof(uchar3)));
	checkCudaErrors(cudaMalloc((void**)&d_out, numPixels * sizeof(uchar3)));
	//make sure no memory is left laying around
	checkCudaErrors(cudaMemset(d_out, 0, numPixels * sizeof(uchar3)));
	// cpy CPU data to GPU data
	checkCudaErrors(cudaMemcpy(d_in, ii->image, numPixels * sizeof(uchar3), cudaMemcpyHostToDevice));

	// launch the kernel
	HW1<<<ii->width, ii->height >>> (d_out, d_in);
	cudaError_t cudaStatus;
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	cudaStatus =  cudaMemcpy(h_out, d_out, numPixels * sizeof(uchar3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	//checkCudaErrors(cudaMemcpy(h_out, d_out, resolution, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_in));
	checkCudaErrors(cudaFree(d_out));
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
	int reselution = width * height;
	ii->height = height;
	ii->width = width;
	ii->resolution = height * width;
	return true;
}

void writeImage(const char* filename, imageInfo* ii, const unsigned char *h_out)
{
	int res = stbi_write_jpg(filename, ii->width, ii->height, 3, h_out, 0);
	if (res == 0)
	{
		std::cout << "Failed to write image file" << std::endl;
		return;
	}
	std::cout << "Write Image Successfully to: " << filename << std::endl;
}

void exec(const char * inputFile, const char * outputFile)
{
	// ¶ÁÈ¡Í¼Æ¬
	imageInfo test;
	imageInfo* ii = &test;
	bool res = readImage(inputFile, ii);
	if (!res) return;
	// ½«Í¼Æ¬»Ò¶È»¯
	unsigned char *h_out = {};
	color2gray(ii,h_out);
	// ±£´æ»Ò¶ÈÍ¼Æ¬
	writeImage(outputFile, ii, h_out);
	stbi_image_free(ii->image);
}