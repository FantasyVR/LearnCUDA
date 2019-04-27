#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void cube(float * d_out, float * d_in);

void getResult(float* h_out, float* h_in,const int ARRAY_BYTES);