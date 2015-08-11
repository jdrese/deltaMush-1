#include <maya/MGlobal.h>
#include <maya/MPxDeformerNode.h>
#include <maya/MTypeId.h> 
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>
#include <vector>
#include <maya/MGlobal.h>

#ifndef _nDeltaMush
#define _nDeltaMush


struct point_data
{
	MIntArray neighbours;
	MVectorArray delta;
	int size;
	double deltaLen;
};

using namespace std;

class nDeltaMush : public MPxDeformerNode
{
public:
	nDeltaMush();
	static  void*		creator();
	static  MStatus		initialize();
	virtual MStatus		deform(MDataBlock& data, MItGeometry& iter, const MMatrix& mat, unsigned int mIndex);
	virtual MStatus     setDependentsDirty( const MPlug& plug, MPlugArray& plugArray );
private:


    void initData( MObject &mesh,
    				int iters );
	
	void averageRelax( MPointArray& source ,
					   MPointArray& target,
					   int iter,
					   double amountV);

	void computeDelta ( MPointArray& source ,
					   MPointArray& target);

	void rebindData(	MObject &mesh,
						int iter,
						double amount
					);


public :
	static MTypeId		id;	
	static MObject		referenceMesh;
	static MObject		rebind ;
	static MObject 		iterations;
	static MObject 		useMulti;
	static MObject		applyDelta;
	static MObject		amount;
	static MObject		mapMult;
    static MObject		globalScale;

private :
	MPointArray targetPos;
	std::vector<point_data> dataPoints;
	bool initialized;


    

};
#endif
