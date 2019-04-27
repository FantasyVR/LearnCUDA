#include "kernal.cuh"

__global__ void cube(float * d_out, float * d_in) {
	// Todo: Fill in this function
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f*f;
}

void getResult(float * h_out, float * h_in, const int ARRAY_BYTES)
{
	float * d_in;
	float * d_out;

	// allocate GPU memory
	cudaMalloc((void**)&d_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	cube << <1, 128 >> > (d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

