#include <maya/MGlobal.h>
#include <maya/MPxDeformerNode.h>
#include <maya/MTypeId.h> 
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>
#include <vector>
#include <maya/MGlobal.h>

#ifndef _DeltaMush
#define _DeltaMush


struct point_data
{
	MIntArray neighbours;
	MVectorArray delta;
	int size;
	double deltaLen;
};

using namespace std;

class DeltaMush : public MPxDeformerNode
{
public:
	DeltaMush();
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
	MPointArray pos;
	std::vector<point_data> dataPoints;
	bool initialized;


    

};
#endif
