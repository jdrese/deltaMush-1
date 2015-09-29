#include <tbb/parallel_for.h>

#include <maya/MPxDeformerNode.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MItGeometry.h>
#include <maya/MItMeshVertex.h>
#include <maya/MVector.h>
#include <maya/MFnMesh.h>
#include <maya/MPointArray.h>
#include <maya/MItMeshPolygon.h>

#include "deltaMush.h"

#if PROFILE ==1
#include <chrono>
using namespace std;
using namespace std::chrono;
#endif

MTypeId     DeltaMush::id( 0x0011FF83); 
const uint DeltaMush::MAX_NEIGH =4;
const uint DeltaMush::GRAIN_SIZE= 2000;
#define SMALL (float)1e-6

MObject DeltaMush::rebind ;
MObject DeltaMush::referenceMesh;
MObject DeltaMush::iterations;
MObject DeltaMush::useMulti;
MObject DeltaMush::applyDelta;
MObject DeltaMush::amount;
MObject DeltaMush::mapMult;
MObject DeltaMush::globalScale;

//I really REALLY hate this, I wish to see if there is some optimization can be done
//here, even VTune shows this to be quite a slow passage, i have some ides in mind like,
//try to hack my way in with memcpy etc, but it would be a hack and would be just for the love
//of performance and not rock solid code
inline void set_matrix_from_vecs(MMatrix &mat,
                            MVector &v1,
                            MVector &v2,
                            MVector &v3)
{
    mat = MMatrix();
    mat[0][0] = v1.x;
    mat[0][1] = v1.y;
    mat[0][2] = v1.z;
    mat[0][3] = 0;
    mat[1][0] = v2.x;
    mat[1][1] = v2.y;
    mat[1][2] = v2.z;
    mat[1][3] = 0;
    mat[2][0] = v3.x;
    mat[2][1] = v3.y;
    mat[2][2] = v3.z;
    mat[2][3] = 0;
    mat[3][0] = 0;
    mat[3][1] = 0;
    mat[3][2] = 0;
    mat[3][3] = 1;
};
DeltaMush::DeltaMush():initialized(false), init(tbb::task_scheduler_init::automatic)
//DeltaMush::DeltaMush():initialized(false), init(20)
{
	targetPos.setLength(0);
    #if PROFILE
    counter =0;
    total =0;
    #endif
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
						uint mIndex )
{	
    #if PROFILE ==1
    auto t1 = high_resolution_clock::now();
    #endif

	//Preliminary check :
	//Check if the ref mesh is connected
	double envelopeV = data.inputValue(envelope).asFloat();
	int iterationsV = data.inputValue(iterations).asInt();
	
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
            MObject referenceMeshV = data.inputValue(referenceMesh).data();
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
        
        //here we invert already the source and targets since the swap happens before computation and not 
        //afterwards, the reason for that is in this way we are always sure that the final result is in the 
        //target pointer
        MPointArray * srcR = &targetPos;
        MPointArray * trgR= &copy;
        int it =0;
        for (it = 0; it < iterationsV; it++)
        {
            swap(srcR, trgR);
            Average_tbb kernel(srcR, trgR, iterationsV ,amountV, neigh_table);
            tbb::parallel_for(tbb::blocked_range<size_t>(0,size,DeltaMush::GRAIN_SIZE), kernel);
        }
        
        if (applyDeltaV >= SMALL )
        {
            Tangent_tbb kernelT (trgR,&pos, applyDeltaV, globalScaleV, envelopeV,
                            wgts, delta_size, delta_table, neigh_table);
            tbb::parallel_for(tbb::blocked_range<size_t>(0,size, DeltaMush::GRAIN_SIZE), kernelT);

            iter.setAllPositions(pos);

        }
        else
        {
            iter.setAllPositions(*trgR);
        }

    }// end of  if (envelopeV > SMALL && iterationsV > 0 ) 
    
    #if PROFILE==1
	auto t2 = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    total += duration;
    counter +=1;
    if (counter >= 100)
    {
        std::cout<<"all deformer: "<<(total/100000.0f)<<" ms"<<std::endl; 
        counter =0;
        total =0;
    }
    #endif
    
    return MStatus::kSuccess ; 
}

Average_tbb::Average_tbb(MPointArray * source ,
					   MPointArray * target , 
                       int iter,
					   double amountV, 
                       const std::vector<int>& neigh_table): 
                       m_source(source), 
                       m_target(target),
                       m_iter(iter), 
                       m_amountV(amountV), 
                       m_neigh_table(neigh_table)
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
            //here we scan the array with a stride of MAX_NEIGH since
            //topology data is stored in a flat buffer, this is particularly
            //good for caching exploit
            ne = m_neigh_table[(i*DeltaMush::MAX_NEIGH) + n];
            temp += (*m_source)[ne];					
        }
        temp/= DeltaMush::MAX_NEIGH;
        //linear interpolation based on amount we want to smooth per iteration
        (*m_target)[i] =(*m_source)[i] +  (temp - (*m_source)[i] )*m_amountV;
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
                const std::vector<int>& neigh_table): 
                m_source(source), 
                m_original(original),
                m_applyDeltaV(applyDeltaV),
                m_envelopeV(envelopeV),
                m_globalScaleV(globalScaleV), 
                m_wgts(wgts), 
                m_delta_size(delta_size),
                m_delta_table(delta_table), 
                m_neigh_table(neigh_table)
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

            v1 = (*m_source)[ m_neigh_table[ne] ] -(*m_source)[i] ;
            v2 = (*m_source)[ m_neigh_table [ne+1] ] -  (*m_source)[i] ;

            v2.normalize();
            v1.normalize();

            cross = v1 ^ v2;
            v2 = cross ^ v1;

            set_matrix_from_vecs(mat, v1, v2, cross);
            delta += (  m_delta_table[ne] * mat );
        }

        //TODO (giordi) the value m_applyDeltaV * m_globalScaleV can be cached and used 
        //rather than be recomputed, also one float only to fit in cache rather than 2
        delta= delta.normal() * m_delta_size[i] * m_applyDeltaV * m_globalScaleV; 
        delta = ((*m_source)[i] + delta) - (*m_original)[i];
        (*m_original)[i]= (*m_original)[i] + (delta * m_wgts[i] * m_envelopeV);
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
	int i , n,ne ;
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
            set_matrix_from_vecs(mat, v1, v2, cross);

            delta_table[ne] =  MVector( delta  * mat.inverse());

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
        tbb::parallel_for(tbb::blocked_range<size_t>(0, size, DeltaMush::GRAIN_SIZE), kernel);
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
