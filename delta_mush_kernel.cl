float3 mat3_vec3_mult(const float3  mx ,
                            const float3  my , 
                            const float3  mz , 
                            const float3  v  
                             )
{
    float3 res;
    res.x = (mx.x * v.x) + (my.x *v.y) + (mz.x *v.z); 
    res.y = (mx.y * v.x) + (my.y *v.y) + (mz.y *v.z); 
    res.z = (mx.z * v.x) + (my.z *v.y) + (mz.z *v.z); 
    return res;
}
__kernel void AverageOpencl(
    __global float* finalPos ,
    __global int * d_neig_table,
    __global const float* initialPos ,
    const uint positionCount
    )
{
    unsigned int positionId = get_global_id(0);
    if ( positionId >= positionCount ) return;

    float3 v = {0.0f,0.0f,0.0f}; 
    int id; 
    for (int i=0; i<4;i++)
    {
       id = d_neig_table[positionId*4 +i]*3;  
       v.x += initialPos[id];
       v.y += initialPos[id+1];
       v.z += initialPos[id+2];
       
    } 
    float FOUR_INV = 1.0f/4.0f;
    v.x *= FOUR_INV;
    v.y *= FOUR_INV;
    v.z *= FOUR_INV;
    vstore3( v, positionId , finalPos );
}

__kernel void TangentSpaceOpencl(
    __global float* finalPos ,
    __global const float* d_delta_table,
    __global const int * d_neig_table,
    __global const float* initialPos ,
    const uint positionCount
    )
{
    unsigned int positionId = get_global_id(0);
    if ( positionId >= positionCount ) return;
    
    //float3 initialPosition = vload3( positionId , initialPos );
    float3 accum = {0.0f,0.0f,0.0f}; 
    float3 v0,v1,v2,crossV,delta,deltaRef;
    unsigned int id; 
    v0 = vload3(positionId,initialPos);
    /*
    if (positionId ==3379)
    {
        printf("as position v0 %f %f %f \n",v0.x,v0.y, v0.z);
    }
    */
    
    for (unsigned int i=0; i<3;i++)
    {
        id = d_neig_table[(positionId*4) +i];  
        v1 = vload3(id, initialPos);
        /*
        if (positionId ==100)
        {
            printf("id %i \n", id);
            printf("v1 %f %f %f \n",v1.x,v1.y, v1.z);
        }
        */ 
        
        id = d_neig_table[positionId*4 +i+1];  
        v2 = vload3(id, initialPos);
        
         
        
         
        v1 -= v0;
        v2 -= v0;
        v1 = normalize(v1);
        v2 = normalize(v2);
        
        
        crossV = cross(v1,v2);
        v2= cross(crossV,v1);

        id = (positionId*9 + (i*3));
        if (positionId ==100)
        {
            printf("id %i \n", id);
            
            //printf("v1 %f %f %f \n",v1.x,v1.y, v1.z);

            //printf("v2 %f %f %f \n",v2.x,v2.y, v2.z);
            //printf("cross %f %f %f \n",crossV.x,crossV.y, crossV.z);
            
        }
        
        deltaRef.x = d_delta_table[id];
        deltaRef.y = d_delta_table[positionId*9 + i*3 +1];
        deltaRef.z = d_delta_table[positionId*9 + i*3 +2];
        
        delta = mat3_vec3_mult(v1, v2, crossV, deltaRef);
        accum += delta;
        if(positionId==100)
        {
           //printf("delta porcaddio \n"); 
            //printf("delta %f %f %f \n",delta.x,delta.y, delta.z);
            //printf("delta reff %f %f %f \n ================ \n",deltaRef.x,deltaRef.y, deltaRef.z);
            //printf("accum %f %f %f \n",accum.x,accum.y, accum.z);
        }
    } 
    accum /= 3.0f; 
    //float3 tmp ={10,10,10};
    accum += v0;
    vstore3( accum, positionId , finalPos );
}

