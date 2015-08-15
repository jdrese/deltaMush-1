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
//make average parallel
//make delta computation parallel
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
            MObject referenceMeshV = data.inputValue(referenceMesh).asMesh();
            pos.setLength(size);	
            targetPos.setLength(size);
            neigh_table.resize(size *MAX_NEIGH);
            delta_table.resize(size *MAX_NEIGH);
            rebindData(referenceMeshV, iterationsV,amountV);


            //read weights
            getWeights(data,size);
            initialized = true;
        }

        iter.allPositions(pos, MSpace::kObject);


        int i,n ;
        MVector delta,v1,v2,cross;

        //float weight;
        MMatrix mat;
        //averageRelax(pos, targetPos, iterationsV, amountV);
         
        copy.copy(pos);
        MPointArray &srcR = copy;
        MPointArray &trgR= targetPos;
        MPointArray &tmp = copy;	
        Average_tbb kernel(srcR, trgR, iterationsV ,amountV, neigh_table);
        int it =0;
        for (it = 0; it < iterationsV; it++)
        {
            tbb::parallel_for(tbb::blocked_range<size_t>(0,size,2000), kernel);
            tmp=srcR;
            srcR = trgR;
            trgR = tmp;


        }
        
        int ne =0;
        int counter =0;
        if (applyDeltaV >= SMALL )
        {

            for (i = 0 ; i <size;i++)
            {
                delta = MVector(0,0,0);
                counter =0;
                for (n = 0; n< MAX_NEIGH -1 ;n++)
                {
                    ne = i*MAX_NEIGH + n; 
                    
                    if (neigh_table[ne] != -1 && neigh_table[ne+1] != -1)
                    {
                        v1 = targetPos[ neigh_table[ne] ] - targetPos[i] ;
                        v2 = targetPos[ neigh_table [ne+1] ] - targetPos[i] ;

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

                        //delta += (  dataPoints[i].delta[n]* mat );
                        delta += (  delta_table[ne]* mat );
                        counter++;
                    }
                }

                delta= (delta/float(counter))*applyDeltaV*globalScaleV; 
                delta = (targetPos[i]+delta) - pos[i];
                pos[i] = pos[i] + (delta * wgts[i] * envelopeV);
            }
        iter.setAllPositions(pos);

        }
        else
        {
            iter.setAllPositions(targetPos);
        }

    }// end of  if (envelopeV > SMALL && iterationsV > 0 ) 

    return MStatus::kSuccess ; 
}

Average_tbb::Average_tbb(MPointArray& source ,
					   MPointArray& target , int iter,
					   double amountV, std::vector<int>& neigh_table): source(source), target(target), tmp(source),iter(iter), amountV(amountV), neigh_table(neigh_table)
{
    
}


void Average_tbb::operator()( const tbb::blocked_range<size_t>& r) const
{

    int i,n,counter,ne;
    MVector temp;
    for (i = r.begin() ; i < r.end() ; i++)
    {
        temp = MVector(0,0,0);
        counter = 0;
        for (n = 0; n<DeltaMush::MAX_NEIGH; n++)
        {
            ne = neigh_table[(i*DeltaMush::MAX_NEIGH) + n];
            if (ne!= -1)
            {
                temp += source[ne];					
                counter +=1;
            }
        }
        temp/= float(counter);
        target[i] =source[i] +  (temp - source[i] )*amountV;
    }

}

void DeltaMush::averageRelax( MPointArray& source ,
					   MPointArray& target , int iter,
					   double amountV)
{
    int size = source.length();
	copy.copy(source);
    
    //initializing references
    MPointArray &srcR = copy;
    MPointArray &trgR= target;
    MPointArray &tmp = copy;	
    
    MVector temp;
	int i , n , it;
    int counter =0;
    int ne =0; 
    for (it = 0; it < iter ; it++)
	{
		for (i = 0 ; i < size ; i++)
		{
			temp = MVector(0,0,0);
			counter = 0;
            for (n = 0; n<MAX_NEIGH; n++)
			{
                ne = neigh_table[(i*MAX_NEIGH) + n];
                if (ne!= -1)
                {
                    temp += srcR[ne];					
                    counter +=1;
                }
			}
			temp/= float(counter);
			trgR[i] =srcR[i] +  (temp - srcR[i] )*amountV;
		}
        tmp=srcR;
        srcR = trgR;
        trgR = tmp;
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
                    neigh_table[(i*MAX_NEIGH)+n] = -1;
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

		//dataPoints[i].deltaLen = delta.length();
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

                    }
		}
	}
}

void DeltaMush::getWeights(MDataBlock data, int size)
{
    MArrayDataHandle inWeightH = data.inputArrayValue(weightList);
    int c = inWeightH.elementCount(); 
    wgts.setLength(size);
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
	MPointArray posRev,back;
	MFnMesh meshFn(mesh);
	meshFn.getPoints(posRev, MSpace::kObject);
	back.copy(posRev);
	averageRelax(posRev , back, iter, amount);
	computeDelta(posRev ,back);
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

