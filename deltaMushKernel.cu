

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>
__global__ void push_kernel(const float * d_in_buffer, float * d_out_buffer, const int size, float amount)
{

    int s_id = ((blockDim.x * blockIdx.x) +threadIdx.x)*3;
    int d_id =  ((blockDim.x * blockIdx.x) +threadIdx.x)*4;
    /*
    if (id>500 && id<600)
    {
        printf("%i \n",id); 
    }
    if (id==552)
    {
        printf("%f %f %f",d_in_buffer[id],d_in_buffer[id+1],d_in_buffer[id+2]);
    }
    
    */
    if(s_id<(size)*3)
    {
       
        d_out_buffer[d_id] = d_in_buffer[s_id] +(1.0f*amount); 
        d_out_buffer[d_id+1] = d_in_buffer[s_id+1] ; 
        d_out_buffer[d_id+2] = d_in_buffer[s_id+2]; 
        d_out_buffer[d_id+3] = 1.0f; 
        
       /*
       d_out_buffer[id] = 1.0f; 
       d_out_buffer[id+1] = 1.0f ; 
       d_out_buffer[id+2] = 1.0f; 
       */
    }
}


float * allocate_buffer(int size, int stride)
{
    float * buffer;
    cudaError_t result;
    result = cudaMalloc((void **) &buffer,stride*size * sizeof(float));
    if (result != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(result));
    return buffer;
}
void kernel_tear_down(float * d_in_buffer, float * d_out_buffer)
{
    if(d_in_buffer);
    {
        cudaFree(d_in_buffer);
        d_in_buffer =0;
    }

    if(d_out_buffer)
    {
        cudaFree(d_out_buffer);
        d_out_buffer=0;
    }
}

void average_launcher(const float * h_in_buffer, float * h_out_buffer, 
                    float * d_in_buffer, float * d_out_buffer, const int size, float amount)
{

    //copy the memory from cpu to gpu
    int buffer_size = 3*size*sizeof(float);
    
    cudaError_t s = cudaMemcpy(d_in_buffer, h_in_buffer, buffer_size, cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying : %s\n", cudaGetErrorString(s));
    
    //setup the kernel
    int grain_size =128;
    size_t width_blocks = ((size%grain_size) != 0)?(size/grain_size) +1: (size/grain_size); 
    dim3 block_size(grain_size,1,1);
    dim3 grid_size(width_blocks,1,1);

    push_kernel<<<grid_size, block_size>>>(d_in_buffer, d_out_buffer, size, amount);
    //cudaDeviceSynchronize();
    //copy data back
    s = cudaMemcpy(h_out_buffer, d_out_buffer, 4*size*sizeof(float), cudaMemcpyDeviceToHost);
    if (s != cudaSuccess) 
            printf("Error copying back: %s\n", cudaGetErrorString(s));
}

