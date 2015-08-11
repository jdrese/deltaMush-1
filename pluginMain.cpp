#include <maya/MGlobal.h>
#include <maya/MFnPlugin.h>

#include "deltaMush.h"


// init
MStatus initializePlugin( MObject obj )
{ 
	MStatus   status;
	MFnPlugin plugin( obj );
    status = plugin.registerNode( "mg_deltaMush", DeltaMush::id, DeltaMush::creator,
                                DeltaMush::initialize, MPxNode::kDeformerNode);

   

	return status;
}
 
MStatus uninitializePlugin( MObject obj)
{
	MStatus   status;
	MFnPlugin plugin( obj );

    status = plugin.deregisterNode( DeltaMush::id );

    
    
   

	return status;
}
