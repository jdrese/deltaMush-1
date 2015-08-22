#include <maya/MGlobal.h>
#include <maya/MFnPlugin.h>

#include "deltaMush.h"
#include "deltaMushOpencl.h"

// init
MStatus initializePlugin( MObject obj )
{ 
	MStatus   status;
	MFnPlugin plugin( obj );
    status = plugin.registerNode( "mg_deltaMush", DeltaMush::id, DeltaMush::creator,
                                DeltaMush::initialize, MPxNode::kDeformerNode);

   
    MString nodeClassName("mg_deltaMush");
    MString registrantId("mg_deltaMushOpencl");
    MGPUDeformerRegistry::registerGPUDeformerCreator(
            nodeClassName,
            registrantId,
            DeltaMushOpencl::getGPUDeformerInfo()
            );
    return status;
}
 
MStatus uninitializePlugin( MObject obj)
{
	MStatus   status;
	MFnPlugin plugin( obj );

    status = plugin.deregisterNode( DeltaMush::id );

    
    
   

	return status;
}
