__kernel void deltaMushOpencl(
    __global float* finalPos ,
    __global int * d_neig_table,
    __global const float* initialPos ,
    const uint positionCount
    )
{
    unsigned int positionId = get_global_id(0);
    if ( positionId >= positionCount ) return;

    float3 initialPosition = vload3( positionId , initialPos );
    
    // Perform some computation to get the final position.
    // ...
    
    float3 finalPosition; 
    finalPosition.x = initialPosition.x +10.0f;
    finalPosition.y = initialPosition.y +10.0f;
    finalPosition.z = initialPosition.z +10.0f;
     
    vstore3( finalPosition , positionId , finalPos );
}
