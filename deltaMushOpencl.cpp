#include "deltaMushOpencl.h"



MGPUDeformerRegistrationInfo* DeltaMushOpencl::getGPUDeformerInfo()
{
    static DeltaMushOpenclInfo theOne;
    return &theOne;
}
DeltaMushOpencl::DeltaMushOpencl()
{
}
DeltaMushOpencl::~DeltaMushOpencl()
{
    terminate();
}
bool DeltaMushOpencl::validateNode(MDataBlock& block, const MEvaluationNode& evaluationNode, const MPlug& plug, MStringArray* messages)
{
    // Support everything.
    return true;
}
MPxGPUDeformer::DeformerStatus DeltaMushOpencl::evaluate(
    MDataBlock& block,
    const MEvaluationNode& evaluationNode,
    const MPlug& plug,
    unsigned int numElements,
    const MAutoCLMem inputBuffer,
    const MAutoCLEvent inputEvent,
    MAutoCLMem outputBuffer,
    MAutoCLEvent& outputEvent
    )
{
    cl_int err = CL_SUCCESS;    
    
    // Setup OpenCL kernel.
    if ( !fKernel.get() )
    {
        // Get and compile the kernel.
        const char* mayaLocation = getenv( "MAYA_LOCATION" );
        MString openCLKernelFile( mayaLocation );
        openCLKernelFile +="/devkit/plug-ins/identityNode/identity.cl";
        MString openCLKernelName("identity");
        MAutoCLKernel kernel = MOpenCLInfo::getOpenCLKernel( openCLKernelFile, openCLKernelName );
        if ( kernel.isNull() )
        {
            return MPxGPUDeformer::kDeformerFailure;
        }
        fKernel = kernel;
        
        // Figure out a good work group size for our kernel.
        fLocalWorkSize = 0;
        fGlobalWorkSize = 0;
        size_t retSize = 0;
        err = clGetKernelWorkGroupInfo(
            fKernel.get(),
            MOpenCLInfo::getOpenCLDeviceId(),
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &fLocalWorkSize,
            &retSize
            );
        MOpenCLInfo::checkCLErrorStatus(err);
        if ( err != CL_SUCCESS || retSize == 0 || fLocalWorkSize == 0)
        {
            return MPxGPUDeformer::kDeformerFailure;
        }
        // Global work size must be a multiple of local work size.
        const size_t remain = numElements % fLocalWorkSize;
        if ( remain )
        {
            fGlobalWorkSize = numElements + ( fLocalWorkSize - remain );
        }
        else
        {
            fGlobalWorkSize = numElements;
        }
    }

        return MPxGPUDeformer::kDeformerSuccess;
}
void DeltaMushOpencl::terminate()
{
    MOpenCLInfo::releaseOpenCLKernel(fKernel);
    fKernel.reset();
}
