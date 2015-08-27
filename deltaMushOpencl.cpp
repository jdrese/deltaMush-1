#include "deltaMushOpencl.h"
#include "deltaMush.h"
#include <maya/MObject.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshVertex.h>
#include <maya/MMatrix.h>
#include <tbb/parallel_for.h>
MString DeltaMushOpencl::pluginLoadPath;
const int DeltaMushOpencl::MAX_NEIGH = 4;
#define SMALL (float)1e-6

MGPUDeformerRegistrationInfo* DeltaMushOpencl::getGPUDeformerInfo()
{
    static DeltaMushOpenclInfo theOne;
    return &theOne;
}
DeltaMushOpencl::DeltaMushOpencl()
{std::cout<<"###########-----------------__##############"<<pluginLoadPath<<std::endl;
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
    // Getting needed data
    float applyDeltaV = block.inputValue(DeltaMush::applyDelta).asDouble();
    float amountV = block.inputValue(DeltaMush::amount).asDouble();
    bool rebindV = block.inputValue(DeltaMush::rebind).asBool();
    float globalScaleV = block.inputValue( DeltaMush::globalScale).asDouble();
    
	double envelopeV = block.inputValue(DeltaMush::envelope).asFloat();
	int iterationsV = block.inputValue(DeltaMush::iterations).asInt();
	
    //MPlug refMeshPlug( thisMObject(), referenceMesh );
    //refMeshPlug.isConnected() 
    if (envelopeV > SMALL && iterationsV > 0  ) 
    {
        cl_int err = CL_SUCCESS;    
        MPxGPUDeformer::DeformerStatus dstatus; 
        // Setup OpenCL kernel.
        if ( !fKernel.get() )
        {
            //opencl boiler plate to setup the kernel
            dstatus = setup_kernel(block, numElements);
            if (dstatus == kDeformerFailure)
            {
                return dstatus;
            }
            //init data builds the neighbour table and we are going to upload it
            MObject referenceMeshV = block.inputValue(DeltaMush::referenceMesh).data();
            //HARDCODED
            int size = numElements;
            m_size = size;
            neigh_table.resize(size *MAX_NEIGH);
            delta_table.resize(size *MAX_NEIGH);
            delta_size.resize(size );
            gpu_delta_table.resize(size *3*(MAX_NEIGH-1));
            rebindData(referenceMeshV, iterationsV,amountV);
            
            //creation and upload
            cl_int clStatus;
            d_neig_table = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY,
                    size*sizeof(int)*MAX_NEIGH, neigh_table.data(),&clStatus);
			MOpenCLInfo::checkCLErrorStatus(clStatus);  
            d_delta_table = clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY,
                    (3*size*(MAX_NEIGH-1)*sizeof(float)), gpu_delta_table.data(),&clStatus);
			MOpenCLInfo::checkCLErrorStatus(clStatus);  
            d_primary=  clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_READ_ONLY,
                    3*size*sizeof(float), NULL,&clStatus);
			MOpenCLInfo::checkCLErrorStatus(clStatus);  
            d_secondary =  clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_READ_ONLY,
                    3*size*sizeof(float), NULL,&clStatus);
			MOpenCLInfo::checkCLErrorStatus(clStatus);  
			int size3 = 3*MAX_NEIGH;
			int size2 = delta_size.size();
            d_delta_size=  clCreateBuffer(MOpenCLInfo::getOpenCLContext(), CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY,
                    size*sizeof(float), delta_size.data(),&clStatus);  

            MOpenCLInfo::checkCLErrorStatus(clStatus);    
        }
        // Set up our input events.  The input event could be NULL, in that case we need to pass
        // slightly different parameters into clEnqueueNDRangeKernel.

        void * src =(void*)&d_primary;
        void * trg =(void*)inputBuffer.getReadOnlyRef(); 

        for (int i=0; i<iterationsV; i++)
        {
            // Set all of our kernel parameters.  Input buffer and output buffer may be changing every frame
            // so always set them.
            swap(src,trg);
            if (i == 1)
            {
                trg = (void*) &d_secondary;
            }
            int ii = i; 
            unsigned int parameterId = 0;
            err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_mem), trg);
            MOpenCLInfo::checkCLErrorStatus(err);
            err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_mem), (void*)&d_neig_table);
            MOpenCLInfo::checkCLErrorStatus(err);
            err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_mem), src);
            MOpenCLInfo::checkCLErrorStatus(err);
            err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_float), (void*)&amountV);
            MOpenCLInfo::checkCLErrorStatus(err);   
            err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_uint), (void*)&ii);
            MOpenCLInfo::checkCLErrorStatus(err);   
            err = clSetKernelArg(fKernel.get(), parameterId++, sizeof(cl_uint), (void*)&numElements);
            MOpenCLInfo::checkCLErrorStatus(err);   

            // Run the kernel

            err = clEnqueueNDRangeKernel(
                    MOpenCLInfo::getOpenCLCommandQueue() ,
                    fKernel.get() ,
                    1 ,
                    NULL ,
                    &fGlobalWorkSize ,
                    &fLocalWorkSize ,
                    0,
                    NULL,
                    NULL 
                    );
            //clWaitForEvents(1,&curr);
            MOpenCLInfo::checkCLErrorStatus(err);
        }
        
        unsigned int parameterId = 0;
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_mem), outputBuffer.getReadOnlyRef());
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_mem), (void*)&d_delta_table);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_mem), (void*)&d_delta_size);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_mem), (void*)&d_neig_table);
        MOpenCLInfo::checkCLErrorStatus(err);
        err = clSetKernelArg(tangent_kernel.get(), parameterId++, sizeof(cl_mem), trg);
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
                0 ,
                NULL ,
                outputEvent.getReferenceForAssignment()

                );
        
        MOpenCLInfo::checkCLErrorStatus(err);

        if ( err != CL_SUCCESS )
        {
            return MPxGPUDeformer::kDeformerFailure;
        }
        
    }    
        return MPxGPUDeformer::kDeformerSuccess;
}
void DeltaMushOpencl::rebindData(		MObject &mesh,
									int iter,
									double amount
								)
{
	initData(mesh );
	MPointArray posRev,back,original;
	MFnMesh meshFn(mesh);
	//building all the arrays
    meshFn.getPoints(posRev, MSpace::kObject);
    back.copy(posRev);
    original.copy(posRev);
    
    //getting ready to kick the parallel kernel 
    int size = posRev.length();
    MPointArray * srcR = &back;
    MPointArray * trgR= &posRev;
    int it =0;
    for (it = 0; it < iter; it++)
    {
        swap(srcR, trgR);
        Average_tbb kernel(srcR, trgR, iter, amount, neigh_table);
        tbb::parallel_for(tbb::blocked_range<size_t>(0,size,2000), kernel);
    }
	computeDelta(original,(*trgR));
}
void DeltaMushOpencl::initData( MObject &mesh)
{
	MFnMesh meshFn(mesh);
	int size = meshFn.numVertices();
    m_size = size;
    neigh_table.resize(size * MAX_NEIGH);
	MItMeshVertex iter(mesh);
	iter.reset();
	
    MIntArray neig_tmp;
    int nsize;
    //looping all the vertices
    for (int i = 0; i < size; i++,iter.next())
    {
        //getting neighbours and their size
        iter.getConnectedVertices(neig_tmp);	
        nsize = neig_tmp.length();
        //if we have more or exactly MAX_NEIGH we flatten the loop
        //and manually put the neighbours indexes in the datastructure
        if (nsize>=MAX_NEIGH)
        {
            neigh_table[i*MAX_NEIGH] = neig_tmp[0];
            neigh_table[(i*MAX_NEIGH)+1] = neig_tmp[1];
            neigh_table[(i*MAX_NEIGH)+2] = neig_tmp[2];
            neigh_table[(i*MAX_NEIGH)+3] = neig_tmp[3];
        } 
        //if not we act differently
        else
        {
            //looping the vertices
            for (int n =0; n<MAX_NEIGH;n++)
            {
                //if n is a valid neighbours we set it otherwise we set again the first neighbour
                //this might need to be fixed in the case we need to set multiple neight in the 
                //else we will end up with matching neighbours, might have to track and bumb and 
                //cycle the index
                if(n<nsize)
                {
                    neigh_table[(i*MAX_NEIGH)+n] = neig_tmp[n];
                } 
                else
                {
                    neigh_table[(i*MAX_NEIGH)+n] = neigh_table[(i*MAX_NEIGH)];
                }
            } //for (int n =0; n<MAX_NEIGH;n++)
        }// if (nsize>=MAX_NEIGH) else
    }// for (int i = 0; i < size; i++,iter.next())
}
void DeltaMushOpencl::computeDelta(MPointArray& source ,
					   MPointArray& target)
{
	int size = source.length();
	MVectorArray arr;
	MVector delta , v1 , v2 , cross;
	int i , n,ne,gpu_id ;
	MMatrix mat;
	//build the matrix
	for ( i = 0 ; i < size ; i++)
	{
		
		delta = MVector ( source[i] - target[i] );
		delta_size[i] = delta.length();
		//get tangent matrices
		for (n = 0; n<MAX_NEIGH-1; n++)
		{
                    ne = i*MAX_NEIGH + n; 
                    
                    if (neigh_table[ne] != -1 && neigh_table[ne+1] != -1)
                    {
			v1 = target[ neigh_table[ne] ] - target[i] ;
			v2 = target[ neigh_table[ne+1] ] - target[i] ;
					 
			v2.normalize();
			v1.normalize();

			cross = v1 ^ v2;
			v2 = cross ^ v1;

			mat = MMatrix();
			mat[0][0] = v1.x;
			mat[0][1] = v1.y;
			mat[0][2] = v1.z;
			mat[0][3] = 0;
			mat[1][0] = v2.x;
			mat[1][1] = v2.y;
			mat[1][2] = v2.z;
			mat[1][3] = 0;
			mat[2][0] = cross.x;
			mat[2][1] = cross.y;
			mat[2][2] = cross.z;
			mat[2][3] = 0;
			mat[3][0] = 0;
            mat[3][1] = 0;
            mat[3][2] = 0;
            mat[3][3] = 1;

            delta_table[ne] =  MVector( delta  * mat.inverse());

            gpu_id = i*9 + n*3; 
            gpu_delta_table[gpu_id] = delta_table[ne][0];
            gpu_delta_table[gpu_id+1] = delta_table[ne][1];
            gpu_delta_table[gpu_id+2] = delta_table[ne][2];
            }
        }
    }
}

MPxGPUDeformer::DeformerStatus DeltaMushOpencl::setup_kernel(MDataBlock& block, int numElements)
{
    cl_int err = CL_SUCCESS;    

    // Get and compile the kernel.

    MString openCLKernelFile(pluginLoadPath);
    openCLKernelFile += "/delta_mush_kernel.cl";
    MString openCLKernelName("AverageOpencl");
    MString openCLKernelNameTan("TangentSpaceOpencl");
    //MString openCLKernelFile("/home/giordi/WORK_IN_PROGRESS/C/deltaMush/delta_mush_kernel.cl");
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
    return MPxGPUDeformer::kDeformerSuccess; 

}

void DeltaMushOpencl::terminate()
{

    
    cl_int err = CL_SUCCESS;    
    err= clReleaseMemObject(d_neig_table);
    MOpenCLInfo::checkCLErrorStatus(err);
    err= clReleaseMemObject(d_delta_table);
    MOpenCLInfo::checkCLErrorStatus(err);
    
    MOpenCLInfo::releaseOpenCLKernel(fKernel);
    fKernel.reset();
}
