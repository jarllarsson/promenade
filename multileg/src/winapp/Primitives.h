#ifndef RAYTRACE_PRIMITIVES_H
#define RAYTRACE_PRIMITIVES_H

// =======================================================================================
//                                      Primitives
// =======================================================================================

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include "RaytraceSurfaceMaterial.h"



using std::vector; 

struct Box
{
	float4 pos;		    // The center point of the box in world coordinates
	float4 sides[3];	// normalized side directions u,v,w
	float hlengths[3]; // positive half-lengths from box center
	Material mat;
	float pad;
};


struct Tri
{
	float4 vertices[3];
	Material mat;
};


struct Plane
{
	float distance;
	float4 normal;
	Material mat;
	float pad[3];
};


struct Sphere
{
	float4 pos;
	float rad;
	Material mat;
	float pad[3];
};


#endif