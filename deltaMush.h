#pragma once

#include <maya/MGlobal.h>
#include <maya/MPxDeformerNode.h>
#include <maya/MTypeId.h> 
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>
#include <vector>
#include <maya/MGlobal.h>
#include <maya/MFloatArray.h>
#include <maya/MEvaluationNode.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <maya/MMatrix.h>
typedef unsigned int uint;
using namespace std;

class DeltaMush : public MPxDeformerNode
{
public:
	DeltaMush();
	static  void*		creator();
	static  MStatus		initialize();
	MStatus		deform(MDataBlock& data, MItGeometry& iter, 
                       const MMatrix& mat, uint mIndex) override;
	MStatus    setDependentsDirty( const MPlug& plug, 
                                    MPlugArray& plugArray ) override;

private:
    void initData( MObject &mesh,
    				int iters );

	void computeDelta ( MPointArray& source ,
					   MPointArray& target);

	void rebindData(	MObject &mesh,
						int iter,
						double amount
					);
    void getWeights(MDataBlock data, int size);
    

    //maya 2016 only overriden functions
    #ifdef Maya2016
    SchedulingType schedulingType()const override;
    MStatus     preEvaluation( const  MDGContext& context, 
                                       const MEvaluationNode& evaluationNode ) override;
    #endif
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
    const static uint MAX_NEIGH;

private :
	MPointArray targetPos;
	MPointArray pos;
	MPointArray copy;
    
    std::vector<float> wgts;
    std::vector<int> neigh_table;
    std::vector<MVector> delta_table;
    std::vector<float> delta_size;
    tbb::task_scheduler_init init;
    
    #if PROFILE==1
    int counter;    
    float total;
    #endif
    bool initialized;

};

//TBB operators
struct Average_tbb
{
    public:
        Average_tbb(MPointArray * source ,
                MPointArray *target , int iter,
                double amountV, const std::vector<int>& neigh_table);

        void operator() (const tbb::blocked_range<size_t>& r)const;

    private:
        MPointArray * source;
        MPointArray * target;
        int iter;
        double amountV;
        const std::vector<int>& neigh_table;

};


struct Tangent_tbb
{
    public:
        Tangent_tbb(MPointArray * source ,
                    MPointArray * original,
                const double applyDeltaV,
                const double globalScaleV,
                const double envelopeV,
                const std::vector<float> & wgts,
                const std::vector<float>& delta_size,
                const std::vector<MVector>& delta_table,
                const std::vector<int>& neigh_table);

        void operator() (const tbb::blocked_range<size_t>& r)const;

    private:
        MPointArray * source;
        MPointArray * original;
        const double applyDeltaV;
        const double envelopeV;
        const double globalScaleV;
        const std::vector<float> & wgts;
        const std::vector<float> & delta_size;
        const std::vector<MVector> &delta_table;
        const std::vector<int>& neigh_table;

};


