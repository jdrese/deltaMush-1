

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdio>


//convert all to use float3 for consistency?
// or float4 to maximize memory bandwith? not sure if applicable
//check for intrinsic SSE operations
//see if is possible to set a float3 in an array index? and set memory of 3 indeces?
__inline__ __device__ float vec_len( float * vec)
{
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

__inline__ __device__ void vec_norm( float * vec)
{
    float len = vec_len(vec);
    vec[0] /= len;
    vec[1] /= len;
    vec[2] /= len;
}
// dot product
inline __device__ float dot(const float * a, const float * b)
{ 
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// cross product
inline __device__ void cross_prod(const float * a, const float * b,float *c )
{ 
    c[0] = (a[1]*b[2]) - (a[2]*b[1]);
    c[1] = (a[2]*b[0]) - (a[0]*b[2]); 
    c[2] = (a[0]*b[1]) - (a[1]*b[0]); 
}

inline __device__ void mat3_vec3_mult(const float * mx ,
                                      const float * my , 
                                      const float * mz , 
                                      const float * v , 
                                      float *res )
{
    res[0] = (mx[0] * v[0]) + (my[0] *v[1]) + (mz[0] *v[2]); 
    res[1] = (mx[1] * v[0]) + (my[1] *v[1]) + (mz[1] *v[2]); 
    res[2] = (mx[2] * v[0]) + (my[2] *v[1]) + (mz[2] *v[2]); 
}
__global__ void average_kernel(float * d_in_buffer, 
                               float * d_out_buffer, 
                               const int * d_neighbours, 
                               const int size, 
                               const float amount)
{
    int s_id = ((blockDim.x * blockIdx.x) +threadIdx.x)*3;
    int d_id =  ((blockDim.x * blockIdx.x) +threadIdx.x)*4;
    
    if(s_id<(size)*3)
    {
        //good case for intrinsinc?
        //__DEVICE_FUNCTIONS_DECL__ unsigned int __vadd4 ( unsigned int  a, unsigned int  b )
        int id;
        float v[3] = {0.0f,0.0f,0.0f};
        float pos[3] = {0.0f,0.0f,0.0f};
        pos[0] = d_in_buffer[s_id];
        pos[1] = d_in_buffer[s_id+1];
        pos[2] = d_in_buffer[s_id+2];
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
        d_out_buffer[s_id] = pos[0] + (v[0]-pos[0]) * amount ; 
        d_out_buffer[s_id+1] = pos[1] + (v[1]-pos[1]) * amount ; 
        d_out_buffer[s_id+2] = pos[2] + (v[2]-pos[2]) * amount ; 
        //d_out_buffer[s_id+1] = v[1]; 
        //d_out_buffer[s_id+2] = v[2]; 
    }
}


__global__ void tangnet_kernel(float * d_smooth, float * d_original, 
                                float * d_delta_table,const int * d_neighbours,
                                float * d_delta_lenghts, 
                                float * d_weights,
                                const int size, const float globalScale,
                                const float envelope, const float applyDelta)

{
    //id stride 3
    int len_id = (blockDim.x * blockIdx.x) +threadIdx.x;
    int s_id = ((blockDim.x * blockIdx.x) +threadIdx.x)*3;
    //id stride 4
    int d_id =  ((blockDim.x * blockIdx.x) +threadIdx.x)*4;
    int delta_id = ((blockDim.x * blockIdx.x) +threadIdx.x)*9;
    
    //local needed variables 
    float v0[3] = {0.0f,0.0f,0.0f};
    float v1[3] = {0.0f,0.0f,0.0f};
    float v2[3] = {0.0f,0.0f,0.0f};
    float cross[3] = {0.0f,0.0f,0.0f};
    float delta[3] = {0.0f,0.0f,0.0f};
    float accum[3] = {0.0f,0.0f,0.0f};
    int id;
    
    if(s_id<(size)*3)
    {
        //central vertex
        v0[0] = d_smooth[s_id]; 
        v0[1] = d_smooth[s_id+1]; 
        v0[2] = d_smooth[s_id+2]; 

        accum[0] =0;
        accum[1] =0;
        accum[2] =0;
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

            vec_norm(&v1[0]);
            vec_norm(&v2[0]);

            cross_prod(v1,v2,cross);
            cross_prod(cross,v1,v2);

            mat3_vec3_mult(v1, v2, cross, &d_delta_table[delta_id +(n*3) ], delta);
            
            accum[0] += delta[0]; 
            accum[1] += delta[1]; 
            accum[2] += delta[2]; 

        }
        vec_norm(accum);
        //float dl =vec_len(&d_delta_table[delta_id ]); 
        accum[0] *= (d_delta_lenghts[len_id] *applyDelta* globalScale); 
        accum[1] *= (d_delta_lenghts[len_id] *applyDelta* globalScale); 
        accum[2] *= (d_delta_lenghts[len_id] *applyDelta* globalScale); 

        float c = d_weights[len_id]* envelope;
        
        d_original[s_id] = (((d_smooth[s_id] + accum[0]) - d_original[s_id])*c) + d_original[s_id]; 
        d_original[s_id+1] = (((d_smooth[s_id+1] + accum[1]) - d_original[s_id+1])*c) + d_original[s_id+1]; 
        d_original[s_id+2] = (((d_smooth[s_id+2] + accum[2]) - d_original[s_id+2])*c) + d_original[s_id+2]; 
        
        }
}
void upload_float(float * h_data, float * d_data,int size)
{
    cudaError_t s = cudaMemcpy(d_data, h_data, size*sizeof(float), cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying : %s\n", cudaGetErrorString(s));

    
}
void upload_int(int * h_data, int * d_data, int size)
{
    cudaError_t s = cudaMemcpy(d_data, h_data, size*sizeof(float), cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying : %s\n", cudaGetErrorString(s));

    
}

void average_launcher(const float * h_in_buffer, float * h_out_buffer, 
                   float * d_in_buffer, float * d_out_buffer, 
                   int * h_neighbours, int* d_neighbours,
                   float * h_delta_table, float * d_delta_table,
                   float * h_delta_lengths, float * d_delta_lenghts,
                   float * h_weights, float * d_weights,
                   const int size,
                   const int iter, 
                   const float amount,
                   const float gloabalScale,
                   const float envelope,
                   const float applyDelta)
{
    cudaEvent_t start, stop;  
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //copy the memory from cpu to gpu
    int buffer_size = 3*size*sizeof(float);

    cudaError_t s = cudaMemcpy(d_in_buffer, h_in_buffer, buffer_size, cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying : %s\n", cudaGetErrorString(s));
    /*
    s = cudaMemcpy(d_neighbours, h_neighbours, 4*size*sizeof(int), cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying neigh_table: %s\n", cudaGetErrorString(s));
    
    s = cudaMemcpy(d_delta_lenghts, h_delta_lengths, size*sizeof(float), cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying neigh_table: %s\n", cudaGetErrorString(s));

    
    s = cudaMemcpy(d_weights, h_weights, size*sizeof(float), cudaMemcpyHostToDevice);
    if (s != cudaSuccess) 
        printf("Error copying neigh_table: %s\n", cudaGetErrorString(s));
    */
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
        average_kernel<<<grid_size, block_size>>>(src, trg, d_neighbours, size,amount);
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
    tangnet_kernel<<<grid_size, block_size>>>(d_out_buffer, d_in_buffer, 
            d_delta_table, d_neighbours,d_delta_lenghts,
            d_weights,size,
            gloabalScale, envelope, applyDelta);

    //copy data back
    s = cudaMemcpy(h_out_buffer, d_in_buffer, 3*size*sizeof(float), cudaMemcpyDeviceToHost);
    if (s != cudaSuccess) 
        printf("Error copying back: %s\n", cudaGetErrorString(s));

    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("cuda computation: %f millisec \n",milliseconds);
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

void kernel_tear_down(float * d_in_buffer, float * d_out_buffer, 
                      int * d_neigh_table, float * d_delta_table,
                      float * d_delta_lenghts, float * d_weights)
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
    if(d_delta_lenghts)
    {
        cudaFree(d_delta_lenghts);
        d_out_buffer = 0;
    }
    if(d_weights)
    {
        cudaFree(d_weights);
        d_out_buffer = 0;
    }
}

