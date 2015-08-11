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

MTypeId     nDeltaMush::id( 0x0011FF83); 

MObject nDeltaMush::rebind ;
MObject nDeltaMush::referenceMesh;
MObject nDeltaMush::iterations;
MObject nDeltaMush::useMulti;
MObject nDeltaMush::applyDelta;
MObject nDeltaMush::amount;
MObject nDeltaMush::mapMult;
MObject nDeltaMush::globalScale;

//MObjectArray nDeltaMush::aWeightMapArray;
//MObjectArray nDeltaMush::aWeightParentArray;

nDeltaMush::nDeltaMush()
{
	initialized = 0 ;
	targetPos.setLength(0);
}

//creator funtion
void* nDeltaMush::creator(){ return new nDeltaMush(); }


MStatus nDeltaMush::initialize()
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

	MGlobal::executeCommand("makePaintable -attrType multiFloat -sm deformer nDeltaMush weights");
	return MStatus::kSuccess;
}

MStatus nDeltaMush::deform( MDataBlock& data, MItGeometry& iter, 
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
  	MObject referenceMeshV = data.inputValue(referenceMesh).asMesh();
	double envelopeV = data.inputValue(envelope).asFloat();
	int iterationsV = data.inputValue(iterations).asInt();
	double applyDeltaV = data.inputValue(applyDelta).asDouble();
	double amountV = data.inputValue(amount).asDouble();
	bool rebindV = data.inputValue(rebind).asBool();
	double globalScaleV = data.inputValue( globalScale).asDouble();


	MPointArray pos;
	iter.allPositions(pos, MSpace::kWorld);

	if (initialized == 0 || rebindV == true)
	{
		rebindData(referenceMeshV, iterationsV,amountV);
		initialized = 1;
	}

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


void nDeltaMush::initData(
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

void nDeltaMush::averageRelax( MPointArray& source ,
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

void nDeltaMush::computeDelta(MPointArray& source ,
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


void nDeltaMush::rebindData(		MObject &mesh,
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

MStatus nDeltaMush::setDependentsDirty( const MPlug& plug, MPlugArray& plugArray )
{
    MStatus status;
	if ( plug == iterations || plug == amount)
    {
		initialized = 0;
	}
    return MS::kSuccess;
}
