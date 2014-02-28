#ifndef RAYTRACE_SCENE_H
#define RAYTRACE_SCENE_H

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include "RaytraceDefines.h"
#include "Primitives.h"
#include "LightStructures.h"
#include "RaytraceSurfaceMaterial.h"


#pragma comment(lib, "cudart") 

using std::vector; 


// =======================================================================================
//                                      Scene
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # Scene
/// 
/// 25-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------


struct Scene
{	
	Light light[AMOUNTOFLIGHTS];
	Sphere sphere[AMOUNTOFSPHERES];
	Plane plane[AMOUNTOFPLANES];
	Tri tri[AMOUNTOFTRIS];
	Box box[AMOUNTOFBOXES];
};

#endif