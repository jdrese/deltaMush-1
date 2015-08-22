#include "deltaMushOpencl.h"
#include "deltaMush.h"
#include <maya/MObject.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshVertex.h>

const int DeltaMushOpencl::MAX_NEIGH = 4;

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
        MString openCLKernelName("AverageOpencl");
        MString openCLKernelNameTan("TangentSpaceOpencl");
        MString openCLKernelFile("/home/giordi/WORK_IN_PROGRESS/C/deltaMush/delta_mush_kernel.cl");
        MAutoCLKernel kernel = MOpenCLInfo::getOpenCLKernel(openCLKernelFile, openCLKernelName );
        tangent_kernel = MOpenCLInfo::getOpenCLKernel(openCLKernelFile, openCLKernelNameTan);

        if ( kernel.isNull() )
        {
            return MPxGPUDeformer::kDeformerFailure;
        }
        fKernel = kernel;
        
        if ( tangent_kernel.isNull() )
        {
            std::cout<<"error getting second kernel from file"<<std::endl;
            return MPxGPUDeformer::kDeformerFailure;
        }
        
        
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
        //init data builds the neighbour table and we are going to upload it
        MObject referenceMeshV = block.inputValue(DeltaMush::referenceMesh).data();
        initData(referenceMeshV);
        //creation and upload
        cl_int clStatus;
        d_neig_table = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY,
                                       m_size*sizeof(int)*MAX_NEIGH, neigh_table.data(),&clStatus);               
        MOpenCLInfo::checkCLErrorStatus(clStatus);    
    }
    // Set up our input events.  The input event could be NULL, in that case we need to pass
    // slightly different parameters into clEnqueueNDRangeKernel.
    cl_event events[ 1 ] = { 0 };
    cl_uint eventCount = 0;
    
    if ( inputEvent.get() )
    {
        events[ eventCount++ ] = inputEvent.get();
    }
    void * src =(void*)outputBuffer.getReadOnlyRef();
    void * trg =(void*)inputBuffer.getReadOnlyRef(); 
    
    for (int i=0; i<20; i++)
    {
        // Set all of our kernel parameters.  Input buffer and output buffer may be changing every frame
        // so always set them.
        swap(src,trg);
        unsigned int parameterId = 0;
        err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_mem), trg);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_mem), (void*)&d_neig_table);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_mem), src);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_uint), (void*)&numElements);
        MOpenCLInfo::checkCLErrorStatus(err);

        // Run the kernel

        MAutoCLEvent temp;
        err = clEnqueueNDRangeKernel(
                MOpenCLInfo::getOpenCLCommandQueue() ,
                fKernel.get() ,
                1 ,
                NULL ,
                &fGlobalWorkSize ,
                &fLocalWorkSize ,
                eventCount ,
                events ,
                temp.getReferenceForAssignment() 
                );
        
        //temp.getReferenceForAssignment() 

        //outputEvent.getReferenceForAssignment()

        MOpenCLInfo::checkCLErrorStatus(err);
    }
    
    //calling tangent space kernel
    //
        unsigned int parameterId = 0;
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_mem), trg);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_mem), (void*)&d_neig_table);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_mem), src);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_uint), (void*)&numElements);
        MOpenCLInfo::checkCLErrorStatus(err);

        // Run the kernel
        err = clEnqueueNDRangeKernel(
                MOpenCLInfo::getOpenCLCommandQueue() ,
                tangent_kernel.get() ,
                1 ,
                NULL ,
                &fGlobalWorkSize ,
                &fLocalWorkSize ,
                eventCount ,
                events ,
                outputEvent.getReferenceForAssignment()

                );
        
        //temp.getReferenceForAssignment() 

        //
        MOpenCLInfo::checkCLErrorStatus(err);

    if ( err != CL_SUCCESS )
    {
        return MPxGPUDeformer::kDeformerFailure;
    }
    return MPxGPUDeformer::kDeformerSuccess;
}

void DeltaMushOpencl::initData(
    			 MObject &mesh)
{
	MFnMesh meshFn(mesh);
	int size = meshFn.numVertices();
    m_size = size;
    neigh_table.resize(size * MAX_NEIGH);
	MPointArray pos,res;
	MItMeshVertex iter(mesh);
	iter.reset();
	//meshFn.getPoints(pos , MSpace::kWorld);
	
    MIntArray neig_tmp;
    int nsize;
	for (int i = 0; i < size; i++,iter.next())
	{
		//point_data pt;
		iter.getConnectedVertices(neig_tmp);	
		nsize = neig_tmp.length();
		//dataPoints[i] = pt;
        if (nsize>=MAX_NEIGH)
        {
           neigh_table[i*MAX_NEIGH] = neig_tmp[0];
           neigh_table[(i*MAX_NEIGH)+1] = neig_tmp[1];
           neigh_table[(i*MAX_NEIGH)+2] = neig_tmp[2];
           neigh_table[(i*MAX_NEIGH)+3] = neig_tmp[3];
        } 
        else
        {
            for (int n =0; n<MAX_NEIGH;n++)
            {
               if(n<nsize)
               {
                    neigh_table[(i*MAX_NEIGH)+n] = neig_tmp[n];
               } 
               else
                {
                    //neigh_table[(i*MAX_NEIGH)+n] = -1;
                    neigh_table[(i*MAX_NEIGH)+n] = neigh_table[(i*MAX_NEIGH)];
                }
            }
        }
	}
}
void DeltaMushOpencl::terminate()
{
    MOpenCLInfo::releaseOpenCLKernel(fKernel);
    fKernel.reset();
}
