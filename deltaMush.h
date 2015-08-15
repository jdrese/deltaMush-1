#include <maya/MGlobal.h>
#include <maya/MPxDeformerNode.h>
#include <maya/MTypeId.h> 
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>
#include <vector>
#include <maya/MGlobal.h>
#include <maya/MFloatArray.h>

#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#ifndef _DeltaMush
#define _DeltaMush

typedef unsigned int uint;

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
    void getWeights(MDataBlock data, int size);


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
    const static unsigned int MAX_NEIGH;

private :
	MPointArray targetPos;
	MPointArray pos;
	MPointArray copy;
    
    MFloatArray wgts;
    std::vector<int> neigh_table;
    std::vector<MVector> delta_table;
    tbb::task_scheduler_init init;
	
    bool initialized;

};
class Average_tbb
{
public:
    /**
    @brief this is the constructor
    @param source: pointer to the source buffer
    @param target: pointer to the targetr buffer
    @param width: the width of the image
    @param height: the height of the image
    */
	Average_tbb(MPointArray& source ,
					   MPointArray& target , int iter,
					   double amountV, std::vector<int>& neigh_table);

    /**
    @brief the () operator called by TBB
    @param r: the range the thread had to work on
    */
	void operator() (const tbb::blocked_range<size_t>& r)const;


private:
    MPointArray& source;
    MPointArray& target;
    MPointArray& tmp;
    int iter;
    double amountV;
    std::vector<int>& neigh_table;

};


#endif


