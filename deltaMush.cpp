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

#define SMALL (float)1e-6

MTypeId     DeltaMush::id( 0x0011FF83); 

MObject DeltaMush::rebind ;
MObject DeltaMush::referenceMesh;
MObject DeltaMush::iterations;
MObject DeltaMush::useMulti;
MObject DeltaMush::applyDelta;
MObject DeltaMush::amount;
MObject DeltaMush::mapMult;
MObject DeltaMush::globalScale;

//TODO
//better average calculation using pointer and not deep copy
//avoid usage of final array and try to use final buffer
//move variable declaration in header and move all attribute pull in pre-load / set dep dirty?
//move the ifs statement that leads to return in one 
//reverse  if statement to make most likely choice as first to help instruction cache
//make average parallel
//make delta computation parallel
//make both average and delta in one parallel call with a lock? faster or not?
//use eigen math library not maya
//use data structures that are not maya (like the arrays)
//trying to enamble eigen SSE instruction or if not possible try to do some intrinsic manually?
//change neighbourood to a max of 4 so we can have a flatter array and we can unroll inner neighbour loop
//if using maya matrix better way to set the value? maybe try to hack a memcpy in ?
//worth considering data refactorying of pos n n n n pos n n n n to ensure cache friendliness? though this is just a guess
//the data refactoring might very well be slower but having a flat buffer of neighbour might help rather then package the point with all the data
DeltaMush::DeltaMush()
{
	initialized = 0 ;
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

	MGlobal::executeCommand("makePaintable -attrType multiFloat -sm deformer DeltaMush weights");
	return MStatus::kSuccess;
}

MStatus DeltaMush::deform( MDataBlock& data, MItGeometry& iter, 
						const MMatrix& localToWorldMatrix, 
						unsigned int mIndex )
{	
	
	//Preliminary check :
	//Check if the ref mesh is connected
	MPlug refMeshPlug( thisMObject(), referenceMesh );
	    
	if (refMeshPlug.isConnected() == false)
	{	
        std::cout<<"ref mesh not connected"<<std::endl;
        return MS::kNotImplemented;
	}

	// Getting needed data
	double envelopeV = data.inputValue(envelope).asFloat();
	int iterationsV = data.inputValue(iterations).asInt();
	double applyDeltaV = data.inputValue(applyDelta).asDouble();
	double amountV = data.inputValue(amount).asDouble();
	bool rebindV = data.inputValue(rebind).asBool();
	double globalScaleV = data.inputValue( globalScale).asDouble();

	if (initialized == 0 || rebindV == true)
	{
  	    MObject referenceMeshV = data.inputValue(referenceMesh).asMesh();
	    pos.setLength(iter.exactCount());	
        rebindData(referenceMeshV, iterationsV,amountV);
		initialized = 1;
	}

	iter.allPositions(pos, MSpace::kWorld);


	if (envelopeV < SMALL ) 
	{
		return MS::kSuccess;
	}

	int size = iter.exactCount();
	int i,n ;
	MVector delta,v1,v2,cross;

	double weight;
	MMatrix mat;
	MPointArray final;
	
    averageRelax(pos, targetPos, iterationsV, amountV);
	if (iterationsV == 0 )
		return MS::kSuccess;
	else
		final.copy(targetPos);
	
	
	for (i = 0 ; i <size;i++)
	{
		delta = MVector(0,0,0);
		if (applyDeltaV >= SMALL )
		{
	
		
			for (n = 0; n<dataPoints[i].size-1; n++)
			{
				v1 = targetPos[ dataPoints[i].neighbours[n] ] - targetPos[i] ;
				v2 = targetPos[ dataPoints[i].neighbours[n+1] ] - targetPos[i] ;
						
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

				delta += (  dataPoints[i].delta[n]* mat );
			}
		}


		delta /= double(dataPoints[i].size);
		delta = delta.normal() * (dataPoints[i].deltaLen*applyDeltaV*globalScaleV); 
		final[i] += delta;

		delta = final[i] - pos[i];
		
		weight = weightValue(data, mIndex, i);
		final[i] = pos[i] + (delta * weight * envelopeV);

	}
	

	iter.setAllPositions(final);

	return MStatus::kSuccess ; 
}


void DeltaMush::initData(
    			 MObject &mesh,
    			 int iters)
{

	MFnMesh meshFn(mesh);
	int size = meshFn.numVertices();

	dataPoints.resize(size);

	MPointArray pos,res;
	MItMeshVertex iter(mesh);
	iter.reset();
	meshFn.getPoints(pos , MSpace::kWorld);
	

	MVectorArray arr;
	for (int i = 0; i < size; i++,iter.next())
	{
		point_data pt;

		iter.getConnectedVertices(pt.neighbours);	
		pt.size = pt.neighbours.length();
		dataPoints[i] = pt;

		arr = MVectorArray();
		arr.setLength(pt.size);
		dataPoints[i].delta = arr;
		
		 
	}

}

void DeltaMush::averageRelax( MPointArray& source ,
					   MPointArray& target , int iter,
					   double amountV)
{


	int size = source.length();
	target.setLength(size);
	
	MPointArray copy;
	copy.copy(source);

	MVector temp;
	int i , n , it;
	for (it = 0; it < iter ; it++)
	{
		for (i = 0 ; i < size ; i++)
		{
			temp = MVector(0,0,0);
			for (n = 0; n<dataPoints[i].size; n++)
			{
					temp += copy[dataPoints[i].neighbours[n]];					
			}

			temp/= double(dataPoints[i].size);
			
			target[i] =copy[i] +  (temp - copy[i] )*amountV;
	
		}
		copy.copy(target);
	}


}

void DeltaMush::computeDelta(MPointArray& source ,
					   MPointArray& target)
{
	int size = source.length();
	MVectorArray arr;
	MVector delta , v1 , v2 , cross;
	int i , n ;
	MMatrix mat;
	//build the matrix
	for ( i = 0 ; i < size ; i++)
	{
		
		delta = MVector ( source[i] - target[i] );

		dataPoints[i].deltaLen = delta.length();
		//get tangent matrices
		for (n = 0; n<dataPoints[i].size-1; n++)
		{
			v1 = target[ dataPoints[i].neighbours[n] ] - target[i] ;
			v2 = target[ dataPoints[i].neighbours[n+1] ] - target[i] ;
					 
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

			dataPoints[i].delta[n] = MVector( delta  * mat.inverse());
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

