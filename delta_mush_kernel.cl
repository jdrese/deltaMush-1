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
    __global float* d_delta_table,
    __global int * d_neig_table,
    __global const float* initialPos ,
    const uint positionCount
    )
{
    unsigned int positionId = get_global_id(0);
    if ( positionId >= positionCount ) return;
    
    float3 initialPosition = vload3( positionId , initialPos );
    float3 accum = {0.0f,0.0f,0.0f}; 
    float3 v0,v1,v2,crossV,delta;
    int id; 
    v0 = vload3(positionId,initialPos);
    for (int i=0; i<3;i++)
    {
        id = d_neig_table[positionId*4 +i];  
        v1 = vload3(id, initialPos);
        id = d_neig_table[positionId*4 +i+1];  
        v2 = vload3(id, initialPos);
    } 
     
    vstore3( v0, positionId , finalPos );
}
