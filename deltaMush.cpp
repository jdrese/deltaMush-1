#include "deltaMush.h"
#include <maya/MPxDeformerNode.h>
#include <maya/MItGeometry.h>
#include <maya/MItMeshVertex.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <maya/MFnMesh.h>
#include <maya/MPointArray.h>
#include <maya/MMatrix.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MPlug.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MFnFloatArrayData.h>

#include <tbb/parallel_for.h>

#define SMALL (float)1e-6

//cuda calls 
#if COMPUTE==1

float * allocate_bufferFloat(int size, int stride);
int * allocate_bufferInt(int size, int stride);
void kernel_tear_down(float * d_in_buffer, float * d_out_buffer, int * d_neigh_table, float * d_delta_table);
void average_launcher(const float * h_in_buffer, float * h_out_buffer, 
                   float * d_in_buffer, float * d_out_buffer, 
                   int * h_neighbours, int * d_neighbours,
                   float * h_delta_table, float * d_delta_table,
                   const int size,int iterationsV);
#endif

MTypeId     DeltaMush::id( 0x0011FF83); 
const unsigned int DeltaMush::MAX_NEIGH =4;

MObject DeltaMush::rebind ;
MObject DeltaMush::referenceMesh;
MObject DeltaMush::iterations;
MObject DeltaMush::useMulti;
MObject DeltaMush::applyDelta;
MObject DeltaMush::amount;
MObject DeltaMush::mapMult;
MObject DeltaMush::globalScale;

//TODO
//better average calculation using pointer and not deep copy X
//avoid usage of final array and try to use final buffer X
//check if possible to avoid normalization and do a simple average =
//move the ifs statement that leads to return in one X 
//reverse  if statement to make most likely choice as first to help instruction cache X
//change neighbourood to a max of 4 so we can have a flatter array and we can unroll inner neighbour loopX
//use data structures that are not maya (like the arrays) X
//move variable declaration in header and move all attribute pull in pre-load / set dep dirty?
//make average parallel X
//make delta computation parallel X
//make both average and delta in one parallel call with a lock? faster or not?
//use eigen math library not maya
//trying to enamble eigen SSE instruction or if not possible try to do some intrinsic manually?
//if using maya matrix better way to set the value? maybe try to hack a memcpy in ?
//worth considering data refactorying of pos n n n n pos n n n n to ensure cache friendliness? though this is just a guess
//the data refactoring might very well be slower but having a flat buffer of neighbour might help rather then package the point with all the data
//possible gpu?
//sorting vertex based on neighbours bucket?

//DeltaMush::DeltaMush():initialized(false), init(tbb::task_scheduler_init::automatic)
DeltaMush::DeltaMush():initialized(false), init(4)
{
    #if COMPUTE==1
    h_out_buffer = nullptr;
    m_cuda_setup =false;
    #endif
    targetPos.setLength(0);
}

//creator funtion
void* DeltaMush::creator(){ return new DeltaMush(); }


MStatus DeltaMush::initialize()
{
	MFnNumericAttribute numericAttr;
	MFnTypedAttribute tAttr;
  
	//globalscale
	globalScale =numericAttr.create("globalScale","gls",MFnNumericData::kDouble,1);
	numericAttr.setKeyable(true);
	numericAttr.setStorable(true);
	numericAttr.setMin(0.0001);
	addAttribute( globalScale );

	rebind = numericAttr.create("rebind", "rbn", MFnNumericData::kBoolean, 0);
	numericAttr.setKeyable(true);
	numericAttr.setStorable(true);
	addAttribute(rebind);

	applyDelta = numericAttr.create("applyDelta", "apdlt", MFnNumericData::kDouble, 1.0);
	numericAttr.setKeyable(true);
	numericAttr.setStorable(true);
	numericAttr.setMin(0);
	numericAttr.setMax(1);
	addAttribute(applyDelta);

	useMulti = numericAttr.create("useMulti", "um", MFnNumericData::kBoolean, 0);
	numericAttr.setKeyable(true);
	numericAttr.setStorable(true);
	//addAttribute(useMulti);

	iterations = numericAttr.create("iterations", "itr", MFnNumericData::kInt, 0);
	numericAttr.setKeyable(true);
	numericAttr.setStorable(true);
	numericAttr.setMin(0);
	addAttribute(iterations);

	amount = numericAttr.create("amount", "am", MFnNumericData::kDouble, 0.5);
	numericAttr.setKeyable(true);
	numericAttr.setStorable(true);
	numericAttr.setMin(0);
	numericAttr.setMax(1);
	addAttribute(amount);

	referenceMesh  = tAttr.create( "referenceMesh", "rfm", MFnData::kMesh );
	tAttr.setKeyable(true);
	tAttr.setWritable(true);
	tAttr.setStorable(true);
	addAttribute(referenceMesh);

	//targets attributes
	mapMult = numericAttr.create("mapMult", "mpm", MFnNumericData::kDouble, 0);
	numericAttr.setArray(true);
	numericAttr.setKeyable(true);
	numericAttr.setWritable(true);
	numericAttr.setStorable(true); 
	numericAttr.setMin(0);
	numericAttr.setMax(1);

	attributeAffects ( referenceMesh , outputGeom);
	attributeAffects ( rebind , outputGeom);
	attributeAffects ( iterations , outputGeom);
	attributeAffects ( applyDelta , outputGeom);
	attributeAffects ( amount, outputGeom);
	attributeAffects ( globalScale, outputGeom);

	MGlobal::executeCommand("makePaintable -attrType multiFloat -sm deformer mg_deltaMush weights");
	return MStatus::kSuccess;
}

MStatus DeltaMush::deform( MDataBlock& data, MItGeometry& iter, 
						const MMatrix& localToWorldMatrix, 
						unsigned int mIndex )
{	
	
	double envelopeV = data.inputValue(envelope).asFloat();
	int iterationsV = data.inputValue(iterations).asInt();
	
    #if COMPUTE==0
    //Preliminary check :
	//Check if the ref mesh is connected
	
    MPlug refMeshPlug( thisMObject(), referenceMesh );
    if (envelopeV > SMALL && iterationsV > 0 && refMeshPlug.isConnected() ) 
    {
        // Getting needed data
        double applyDeltaV = data.inputValue(applyDelta).asDouble();
        double amountV = data.inputValue(amount).asDouble();
        bool rebindV = data.inputValue(rebind).asBool();
        double globalScaleV = data.inputValue( globalScale).asDouble();

        int size = iter.exactCount();
        if (initialized == false || rebindV == true)
        {
            MObject referenceMeshV = data.inputValue(referenceMesh).asMesh();
            pos.setLength(size);	
            targetPos.setLength(size);
            neigh_table.resize(size *MAX_NEIGH);
            delta_table.resize(size *MAX_NEIGH);
            delta_size.resize(size );
            rebindData(referenceMeshV, iterationsV,amountV);


            //read weights
            getWeights(data,size);
            initialized = true;
        }

        iter.allPositions(pos, MSpace::kObject);

        //We need to work on a copy due to the fact that we need to preserve the original position 
        //for blending afterwards 
        
        copy.copy(pos);
        //here we invert already the source and targets since the swap appens before computtion and not 
        //afterwards, the reason for that is in this way we are always sure that the final result is in the 
        //target pointer
        MPointArray * srcR = &targetPos;
        MPointArray * trgR= &copy;
        int it =0;
        for (it = 0; it < iterationsV; it++)
        {
            swap(srcR, trgR);
            Average_tbb kernel(srcR, trgR, iterationsV ,amountV, neigh_table);
            tbb::parallel_for(tbb::blocked_range<size_t>(0,size,2000), kernel);
        }
        
        if (applyDeltaV >= SMALL )
        {
            Tangent_tbb kernelT (trgR,&pos, applyDeltaV, globalScaleV, envelopeV,
                            wgts, delta_size, delta_table, neigh_table);
            tbb::parallel_for(tbb::blocked_range<size_t>(0,size,2000), kernelT);

            iter.setAllPositions(pos);

        }
        else
        {
            iter.setAllPositions(*trgR);
        }

    }// end of  if (envelopeV > SMALL && iterationsV > 0 ) 
    #else
    
    MPlug refMeshPlug( thisMObject(), referenceMesh );
    if (envelopeV > SMALL && iterationsV > 0 && refMeshPlug.isConnected() ) 
    {
        int i=0;

        //CUDA

        MArrayDataHandle inMeshH= data.inputArrayValue( input ) ;
        inMeshH.jumpToArrayElement( 0 ) ;
        MObject inMesh= inMeshH.inputValue().child( inputGeom ).asMesh() ;
        MFnMesh meshFn(inMesh) ;

        //float am = data.inputValue(amount).asFloat();
        float am = data.inputValue(amount).asDouble();

        MStatus stat;
        const float * v_data = meshFn.getRawPoints(&stat);
        const int size = iter.exactCount(); 
        double applyDeltaV = data.inputValue(applyDelta).asDouble();
        double amountV = data.inputValue(amount).asDouble();
        bool rebindV = data.inputValue(rebind).asBool();
        double globalScaleV = data.inputValue( globalScale).asDouble();
        if (initialized == false || rebindV == true)
        {
            MObject referenceMeshV = data.inputValue(referenceMesh).asMesh();
            pos.setLength(size);	
            targetPos.setLength(size);
            neigh_table.resize(size *MAX_NEIGH);
            delta_table.resize(size *MAX_NEIGH);
            //this one is a flat array of floats, means we are gonna
            //store 12 floats for each vertex
            
            gpu_delta_table.resize(size *3*(MAX_NEIGH-1));

            delta_size.resize(size );
            rebindData(referenceMeshV, iterationsV,amountV);


            //read weights
            getWeights(data,size);
            initialized = true;
        }


        if(!m_cuda_setup)
        {
            std::cout<<"setting cuda stuff"<<std::endl;
            d_in_buffer = allocate_bufferFloat(size,3);
            d_out_buffer = allocate_bufferFloat(size,3);
            d_neighbours= allocate_bufferInt(size,MAX_NEIGH);
            d_delta_table= allocate_bufferFloat(size,9);
            h_out_buffer = new float[3*size]; 
            m_cuda_setup= true;
        }

        average_launcher(v_data, h_out_buffer, 
                d_in_buffer, d_out_buffer, 
                neigh_table.data(), d_neighbours,
                gpu_delta_table.data(), d_delta_table,
                size, iterationsV);


        MPointArray outp;
        outp.setLength(size);

        MPoint tmp;
        int c=0; 
        for (int i=0; i<size*3;i+=3,c++)
        {
            tmp = MPoint((float)h_out_buffer[i],
                        (float)h_out_buffer[i+1],
                        (float)h_out_buffer[i+2],1.0f);
            outp[c] =tmp ;
        }
        iter.setAllPositions(outp);

    }
    #endif
    
    return MStatus::kSuccess ; 
}

DeltaMush::~DeltaMush()
{
    #if COMPUTE==1
    kernel_tear_down(d_in_buffer, d_out_buffer, d_neighbours, d_delta_table);
    if(h_out_buffer)
    {
        delete(h_out_buffer);
    }
    #endif
}


Average_tbb::Average_tbb(MPointArray * source ,
					   MPointArray * target , int iter,
					   double amountV, const std::vector<int>& neigh_table): source(source), target(target),iter(iter), amountV(amountV), neigh_table(neigh_table)
{
    
}


void Average_tbb::operator()( const tbb::blocked_range<size_t>& r) const
{

    int i,n,ne;
    MVector temp;
    for (i = r.begin() ; i < r.end() ; i++)
    {
        temp = MVector(0,0,0);
        for (n = 0; n<DeltaMush::MAX_NEIGH; n++)
        {
            ne = neigh_table[(i*DeltaMush::MAX_NEIGH) + n];
            //need to work on this if, find a way to remove it
                temp += (*source)[ne];					
        }
        temp/= DeltaMush::MAX_NEIGH;
        (*target)[i] =(*source)[i] +  (temp - (*source)[i] )*amountV;
    }

}

Tangent_tbb::Tangent_tbb(MPointArray * source ,
                MPointArray * original,
                const double applyDeltaV,
                const double envelopeV,
                const double globalScaleV,
                const std::vector<float> & wgts,
                const std::vector<float> & delta_size,
                const std::vector<MVector> & delta_table,
                const std::vector<int>& neigh_table): source(source), original(original),
                                                        applyDeltaV(applyDeltaV),envelopeV(envelopeV),
                                                      globalScaleV(globalScaleV), wgts(wgts), delta_size(delta_size),
                                                      delta_table(delta_table), neigh_table(neigh_table)
{}

void Tangent_tbb::operator()( const tbb::blocked_range<size_t>& r) const
{
    int i,n,ne;
    MVector delta,v1,v2,cross;
    MMatrix mat;
    for (i = r.begin() ; i <r.end();i++)
    {
        delta = MVector(0,0,0);
        for (n = 0; n< DeltaMush::MAX_NEIGH -1 ;n++)
        {
            ne = i*DeltaMush::MAX_NEIGH + n; 

            //if (neigh_table[ne] != -1 && neigh_table[ne+1] != -1)
            //{
                v1 = (*source)[ neigh_table[ne] ] -(*source)[i] ;
                v2 = (*source)[ neigh_table [ne+1] ] -  (*source)[i] ;

                v2.normalize();
                v1.normalize();

                cross = v1 ^ v2;
                v2 = cross ^ v1;
                if(i == 100 && n==0)
                {
                    std::cout<<v1<<std::endl;
                    std::cout<<v2<<std::endl;
                    std::cout<<cross<<std::endl;
                }


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

                delta += (  delta_table[ne]* mat );
            //}
        }

        //delta= (delta/DeltaMush::MAX_NEIGH)*applyDeltaV*globalScaleV; 
        delta= delta.normal()*delta_size[i]*applyDeltaV*globalScaleV; 
        delta = ((*source)[i]+delta) - (*original)[i];
        (*original)[i]= (*original)[i] + (delta * wgts[i] * envelopeV);
    }


}

void DeltaMush::initData(
    			 MObject &mesh,
    			 int iters)
{

	MFnMesh meshFn(mesh);
	int size = meshFn.numVertices();

	MPointArray pos,res;
	MItMeshVertex iter(mesh);
	iter.reset();
	meshFn.getPoints(pos , MSpace::kWorld);
	
	MVectorArray arr;
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

void DeltaMush::computeDelta(MPointArray& source ,
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
                #if COMPUTE == 1
                gpu_id = i*9 + n*3; 
                gpu_delta_table[gpu_id] = delta_table[ne][0];
                gpu_delta_table[gpu_id+1] = delta_table[ne][1];
                gpu_delta_table[gpu_id+2] = delta_table[ne][2];
                #endif
            }
        }
    }
}

void DeltaMush::getWeights(MDataBlock data, int size)
{
    MArrayDataHandle inWeightH = data.inputArrayValue(weightList);
    int c = inWeightH.elementCount(); 
    wgts.resize(size);
    //if there is no map we initialize it to 1.0
    if (c>0)
    { 
       MStatus stat = inWeightH.jumpToElement(0);
       
       MDataHandle targetListFirst= inWeightH.inputValue(&stat);
       MDataHandle weightFirst= targetListFirst.child(weights); 
       MArrayDataHandle weightArrH(weightFirst);
       int elem = weightArrH.elementCount();
       weightArrH.jumpToElement(0);
       
       for(int i=0; i<elem; i++, weightArrH.next())
       {
            wgts[i] = weightArrH.inputValue().asFloat();
       }

    }
    else
    {
       for(int i=0; i<size; i++ )
       {
            wgts[i] = 1.0f;

       }
    }
       
}
void DeltaMush::rebindData(		MObject &mesh,
									int iter,
									double amount
								)
{
	initData(mesh , iter);
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

MStatus DeltaMush::setDependentsDirty( const MPlug& plug, MPlugArray& plugArray )
{
    MStatus status;
	if ( plug == iterations || plug == amount)
    {
		initialized = 0;
	}
    return MS::kSuccess;
}

#ifdef Maya2016
DeltaMush::SchedulingType DeltaMush::schedulingType()const
{
    return SchedulingType::kParallel;
}

MStatus DeltaMush::preEvaluation( const  MDGContext& context, const MEvaluationNode& evaluationNode )
{
    MStatus status;
    if( ( evaluationNode.dirtyPlugExists(iterations, &status) && status ) ||  
            ( evaluationNode.dirtyPlugExists(amount, &status) && status ) )
    {
        initialized=0;
    }
    return MS::kSuccess;


}

#endif
