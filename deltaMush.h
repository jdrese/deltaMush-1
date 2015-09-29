#pragma once

#include <vector>

#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>

#include <maya/MGlobal.h>
#include <maya/MPxDeformerNode.h>
#include <maya/MTypeId.h> 
#include <maya/MPointArray.h>
#include <maya/MMatrix.h>
#include <maya/MEvaluationNode.h>

typedef unsigned int uint;
using namespace std;

/*
 * This deformer is a deformer which performs a smooth and computes at bind pose the amount of inflation 
 * the mesh gets, at run time this offest is applied back in tangent space in order to preserve volume
 */

class DeltaMush : public MPxDeformerNode
{
public:
	/*
     * The conostructor
     */
    DeltaMush();

    /* 
     * The creator function in charge of return a new node pointer
     */
	static  void*		creator();
	
    /*
     * This funciton is in charge of building the needed ports on the node
     */
    static  MStatus		initialize();

    /*
     * This is the core function that gets called per node evaluation
     * and where the heavy computation lies, args not documented since is a maya overriden function
     */
	MStatus	deform(MDataBlock& data, MItGeometry& iter, 
                       const MMatrix& mat, uint mIndex) override;
	
    /*
     * Set dependents dirty function call is used to perform checks on what plugs
     * are dirty in order to set interna flags to recompute the minimum amount possible
     * of data, arguments not documented since is an overidden maya funciton
     */
    MStatus setDependentsDirty( const MPlug& plug, 
                                    MPlugArray& plugArray ) override;

private:
    /*
     * This function is in charge of computing the deltas representing the loss of volume,
     * this function gets colled at bind time only
     * @param mesh: reference to an MObject containing the mesh data
     * @param iters: number of smooth iteration to perform
     */
    void initData( MObject &mesh,
    				int iters );

	/*
     * This sub-funciton computes the actual delta between two given meshes
     * @param source: Array of points representing the reference mesh
     * @param target: Array of smoothed points
     */
    void computeDelta ( MPointArray& source ,
					    MPointArray& target);

	/*
     * This function gets called whenever any of the inputs is changed 
     * performing the needed caching of the static data
     * @param mesh: mesh to smoooth
     * @param int: number of iteration for the smooth
     * @param double: amount of smooth applied at each iteration (between 0 and 1)
     */
    void rebindData(	MObject &mesh,
						int iter,
						double amount
					);
    /* 
     * Function that pulls all the weights at once
     * @param data: the deformer data provided from maya
     * @param size: the number of vertex in the mesh
     */
    void getWeights(MDataBlock data, int size);
    

    //maya 2016 only overriden functions, those function are needed to trigger
    //parallel evaluation of DAG in the new evaluation system.
    #ifdef Maya2016
    SchedulingType schedulingType()const override;
    MStatus     preEvaluation( const  MDGContext& context, 
                                       const MEvaluationNode& evaluationNode ) override;
    #endif
public :
    //Maya static atributes
	static MTypeId		id;	
	static MObject		referenceMesh;
	static MObject		rebind ;
	static MObject 		iterations;
	static MObject 		useMulti;
	static MObject		applyDelta;
	static MObject		amount;
	static MObject		mapMult;
    static MObject		globalScale;
    //Constant value for maximum number of vertexes
    const static uint MAX_NEIGH;
    const static uint GRAIN_SIZE;

private :
    //Variables holding the points mesh data
	MPointArray targetPos;
	MPointArray pos;
	MPointArray copy;
    
    //weights data
    std::vector<float> wgts;
    //topology table of the mesh
    std::vector<int> neigh_table;
    //buffer holding the vectors representing volume loss deltas
    std::vector<MVector> delta_table;
    //original size of the deltas
    std::vector<float> delta_size;
    //TBB scheduler
    tbb::task_scheduler_init init;
    
    #if PROFILE==1
    int counter;    
    float total;
    #endif
    //bool for chekcing if node has been initialized or not
    bool initialized;

};

//TBB operators

//This funcntors is in charge to kick in parallel the smoothing pass of the mess
struct Average_tbb
{
    public:
        /*
         * The constructor
         * @param source: the mesh to smooth
         * @param target: target buffer we are gonna write to
         * @param iter: the number of smooth iterations
         * @param amountV: the amount applied for each smooth pass
         * @prama neigh_table: topology data of the mesh
         */
        Average_tbb(MPointArray * source ,
                MPointArray *target , int iter,
                double amountV, const std::vector<int>& neigh_table);

        /* 
         * Functor operator called by TBB
         */
        void operator() (const tbb::blocked_range<size_t>& r)const;

    private:
        //cached inputs
        MPointArray * m_source;
        MPointArray * m_target;
        int m_iter;
        double m_amountV;
        const std::vector<int>& m_neigh_table;

};

//this functor deforms the offest in tangent space
struct Tangent_tbb
{
    public:
        /*
         * The constructor
         * @param source: the currently deformed mesh
         * @param original: the original data set
         * @param applyDeltaV: the amount of delta to apply
         * @param globalScaleV: the value of global scale of the mesh
         * @param envelopV: amount of the global deformed that gets applied
         * @param wgts: the weight map for the deformer
         * @param delta_table:  the deltas to be transformed in tangent space 
         * @param neigh_table:  the topology table of the mes
         */
        Tangent_tbb(MPointArray * source ,
                    MPointArray * original,
                const double applyDeltaV,
                const double globalScaleV,
                const double envelopeV,
                const std::vector<float> & wgts,
                const std::vector<float>& delta_size,
                const std::vector<MVector>& delta_table,
                const std::vector<int>& neigh_table);

        /* 
         * Functor operator called by TBB
         */
        void operator() (const tbb::blocked_range<size_t>& r)const;

    private:
        //cached inputs
        MPointArray * m_source;
        MPointArray * m_original;
        const double m_applyDeltaV;
        const double m_envelopeV;
        const double m_globalScaleV;
        const std::vector<float>& m_wgts;
        const std::vector<float>& m_delta_size;
        const std::vector<MVector>& m_delta_table;
        const std::vector<int>& m_neigh_table;

};


