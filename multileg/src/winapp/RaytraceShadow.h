#ifndef RAYTRACE_SHADOW_H
#define RAYTRACE_SHADOW_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include "KernelMathHelper.h"
#include "LightStructures.h"
#include "IntersectionInfo.h"
#include "IntersectAll.h"

__device__ float ShadowCastAll(const Light* in_light,
							   int shadowMode, 
							   Ray* inout_shadowRay,
							   Intersection* inout_intersection,
							   const Scene* in_scene,
							   float u,
							   float v,
							   float partLit)
{

	float lightHits=0.0f;
	inout_intersection->dist = MAX_INTERSECT_DIST; // reset
	inout_shadowRay->origin = inout_intersection->pos;

	if (shadowMode>RAYTRACESHADOWMODE_HARD) // soft shadow
	{
		for (int shadows=0;shadows<shadowMode;shadows++)
		{
			float3 spread = cu_getRandomVector3(make_float2(u+(float)shadows,v+(float)shadows))*0.02f;
			inout_shadowRay->dir = cu_normalize( inout_intersection->pos * in_light->vec.w - in_light->vec + make_float4(spread.x,spread.y,spread.z,0.0f));
			if (!IntersectAll(in_scene,inout_shadowRay,inout_intersection,true)) // only add light colour if no object is between current pixel and light*/
				lightHits+=partLit;
		}	
	}	
	else if (shadowMode==RAYTRACESHADOWMODE_HARD) // hard shadow
	{
		inout_shadowRay->dir = cu_normalize( inout_intersection->pos * in_light->vec.w - in_light->vec );
		if (!IntersectAll(in_scene,inout_shadowRay,inout_intersection,true)) // only add light colour if no object is between current pixel and light*/
			lightHits=1.0f;
	}

	return lightHits;
}

#endif