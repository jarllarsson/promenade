#ifndef INTERSECTIONINFO_H
#define INTERSECTIONINFO_H

#include "RaytraceSurfaceMaterial.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


struct Intersection
{
	float4 normal;
	float4 pos;
	Material surface; // don't allow changing the material's values from intersection struct
	float dist;
};


#endif