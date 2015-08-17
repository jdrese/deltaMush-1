

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>
__global__ void average_kernel(float * d_in_buffer, float * d_out_buffer, 
                            const int * d_neighbours, const int size)
{
    int s_id = ((blockDim.x * blockIdx.x) +threadIdx.x)*3;
    int d_id =  ((blockDim.x * blockIdx.x) +threadIdx.x)*4;
    
    if(s_id<(size)*3)
    {
        int id;
        float v[3] = {0.0f,0.0f,0.0f};
        for (int i=0; i<4;i++)
        {
            id = d_neighbours[d_id+i]*3;
            v[0] += d_in_buffer[id]; 
            v[1] += d_in_buffer[id+1]; 
            v[2] += d_in_buffer[id+2]; 
        }
        v[0]/= 4.0f;
        v[1]/= 4.0f;
        v[2]/= 4.0f;
        d_out_buffer[s_id] = v[0]; 
        d_out_buffer[s_id+1] = v[1]; 
        d_out_buffer[s_id+2] = v[2]; 
    }
}

__global__ void tangnet_kernel(float * d_smooth, float * d_original, float * d_delta_table,const int * d_neighbours, const int size)

{
    //id stride 3
    int s_id = ((blockDim.x * blockIdx.x) +threadIdx.x)*3;
    //id stride 4
    int d_id =  ((blockDim.x * blockIdx.x) +threadIdx.x)*4;
    
    //local needed variables 
    float v0[3] = {0.0f,0.0f,0.0f};
    float v1[3] = {0.0f,0.0f,0.0f};
    float v2[3] = {0.0f,0.0f,0.0f};
    float cross[3] = {0.0f,0.0f,0.0f};
    int id;
    
    if(s_id<(size)*3)
    {
        //central vertex
        v0[0] = d_smooth[s_id]; 
        v0[1] = d_smooth[s_id]; 
        v0[2] = d_smooth[s_id]; 

        for (int n=0; n<3;n++)
        {

            id = d_neighbours[d_id+n]*3;
            //first neighbour position
            v1[0] = d_smooth[id]; 
            v1[1] = d_smooth[id+1]; 
            v1[2] = d_smooth[id+2]; 
            
            id = d_neighbours[d_id+n+1]*3;
            //second neighbour position
            v2[0] = d_smooth[id]; 
            v2[1] = d_smooth[id+1]; 
            v2[2] = d_smooth[id+2]; 

            //generate proper vectors
            v1[0] -= v0[0];
            v1[1] -= v0[1];
            v1[2] -= v0[2];
            
            v2[0] -= v0[0];
            v2[1] -= v0[1];
            v2[2] -= v0[2];
            
        }
    }
}

__inline__ __device__ void vec_norm( float * vec)
{

}

void average_launcher(const float * h_in_buffer, float * h_out_buffer, 
                   float * d_in_buffer, float * d_out_buffer, 
                   int * h_neighbours, int* d_neighbours,
                   float * h_delta_table, float * d_delta_table,
                   const int size,int iter)
{
    //copy the memory from cpu to gpu
    int buffer_size = 3*size*sizeof(float);
    
    cudaError_t s = cudaMemcpy(d_in_buffer, h_in_buffer, buffer_size, cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying : %s\n", cudaGetErrorString(s));
    
    s = cudaMemcpy(d_neighbours, h_neighbours, 4*size*sizeof(int), cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying neigh_table: %s\n", cudaGetErrorString(s));
    
    //setup the kernel
    int grain_size =128;
    size_t width_blocks = ((size%grain_size) != 0)?(size/grain_size) +1: (size/grain_size); 
    dim3 block_size(grain_size,1,1);
    dim3 grid_size(width_blocks,1,1);
    
    float * trg= d_in_buffer;
    float * src= d_out_buffer; 
    float * tmp;
    for (int i =0; i<iter; i++)
    {
        tmp = src;
        src = trg;
        trg =tmp; 
        average_kernel<<<grid_size, block_size>>>(src, trg, d_neighbours, size);
    }

    //copy  original data back up
    //if i run the above thread async i might be able to kick this extra memcpy already?
    //to do so I might need another buffer tho
    s = cudaMemcpy(d_in_buffer, h_in_buffer, buffer_size, cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying : %s\n", cudaGetErrorString(s));
    //upload deltas 
    s = cudaMemcpy(d_delta_table, h_delta_table, 9*size*sizeof(float), cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying : %s\n", cudaGetErrorString(s));
    tangnet_kernel<<<grid_size, block_size>>>(d_out_buffer, d_in_buffer, d_delta_table, d_neighbours,size);
    


    //copy data back
    s = cudaMemcpy(h_out_buffer, d_out_buffer, 3*size*sizeof(float), cudaMemcpyDeviceToHost);
    if (s != cudaSuccess) 
            printf("Error copying back: %s\n", cudaGetErrorString(s));
}


float * allocate_bufferFloat(int size, int stride)
{
    float * buffer;
    cudaError_t result;
    result = cudaMalloc((void **) &buffer,stride*size * sizeof(float));
    if (result != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(result));
    return buffer;
}
int * allocate_bufferInt(int size, int stride)
{
    int * buffer;
    cudaError_t result;
    result = cudaMalloc((void **) &buffer,stride*size * sizeof(int));
    if (result != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(result));
    return buffer;
}

void kernel_tear_down(float * d_in_buffer, float * d_out_buffer, int * d_neigh_table, float * d_delta_table)
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
    
    if(d_neigh_table)
    {
        cudaFree(d_neigh_table);
        d_out_buffer = 0;
    }
    if(d_delta_table)
    {
        cudaFree(d_delta_table);
        d_out_buffer = 0;
    }
}

