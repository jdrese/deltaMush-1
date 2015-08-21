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
#ifndef _DeltaMush
#define _DeltaMush

typedef unsigned int uint;

using namespace std;

class DeltaMush : public MPxDeformerNode
{
public:
	DeltaMush();
    ~DeltaMush();
	static  void*		creator();
	static  MStatus		initialize();
	virtual MStatus		deform(MDataBlock& data, MItGeometry& iter, const MMatrix& mat, unsigned int mIndex);
	virtual MStatus     setDependentsDirty( const MPlug& plug, MPlugArray& plugArray );
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

    #ifdef Maya2016
    virtual SchedulingType schedulingType()const;
    virtual MStatus     preEvaluation( const  MDGContext& context, const MEvaluationNode& evaluationNode );
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
    const static unsigned int MAX_NEIGH;

private :
	MPointArray targetPos;
	MPointArray pos;
	MPointArray copy;
    
    unique_ptr<float[]> v_data;
    std::vector<float> wgts;
    std::vector<int> neigh_table;
    std::vector<MVector> delta_table;
    std::vector<float> delta_size;
    tbb::task_scheduler_init init;
    bool initialized;
    //cuda stuff
    bool m_cuda_setup;
    float * h_out_buffer;
    float * d_out_buffer;
    float * d_in_buffer;
    int* d_neighbours;
    float * d_delta_table;
    float * d_delta_lenghts;
    float * d_weights;
    std::vector<float> gpu_delta_table;
    MPointArray outp;
};
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

struct MPointArrayToBuffer
{
    public:
        MPointArrayToBuffer(MPointArray & parr,
                            float * buffer);
        void operator() (const tbb::blocked_range<size_t>& r)const;
    private:
        MPointArray & m_parr;
        float * m_buffer;

};

struct BufferToMPointArray
{
    public:
        BufferToMPointArray( MPointArray & parr,
                            float * buffer);
        void operator() (const tbb::blocked_range<size_t>& r)const;
    private:
        MPointArray & m_parr;
        float * m_buffer;

};
#endif


