#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>


float * allocate_buffer(int size, int stride)
{
    float * buffer;
    cudaError_t result;
    result = cudaMalloc((void **) &buffer,stride*size * sizeof(float));
    if (result != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(result));
    return buffer;
}

