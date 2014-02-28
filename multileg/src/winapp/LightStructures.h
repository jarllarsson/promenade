#ifndef RAYTRACE_LIGHTSTRUCTURES_H
#define RAYTRACE_LIGHTSTRUCTURES_H


// =======================================================================================
//                                    Light Structures
// =======================================================================================

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

using std::vector; 


struct SurfaceLightingData
{
	float4 diffuseColor;
	float4 specularColor;
};

struct Light
{
	float diffusePower;
	float specularPower;
    float4 vec; // direction(w=0) or position(w=1)
	float4 diffuseColor;
	float4 specularColor;
};

#endif