#ifndef INTERSECT_BOX_H
#define INTERSECT_BOX_H

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>


#include "KernelMathHelper.h"
#include "RaytraceLighting.h"
#include "Primitives.h"
#include "Ray.h"


using std::vector; 

#define BOXEPSILON 1.0f/100000.0f

__device__ bool IntersectBox(const Box* in_box, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	float tMin=-FLT_MAX;
	float tMax=FLT_MAX;
	float4 pos = in_box->pos-in_ray->origin;
	int i=0, hitSideMin=0, hitSideMax=0; 
	float hitMinSign=1, hitMaxSign=1;
	for (i=0;i<3;i++) // for each side of the box
	{		
		// check too see whether the ray is perpendicular to the normal
		// direction of the current slab
		float e = cu_dot(in_box->sides[i], pos);
		float f = cu_dot(in_box->sides[i], in_ray->dir);	
		float abs_f = fabs(f);
		if (abs_f > BOXEPSILON)
		{
			float m  = 1.0f/f;	// calculate 1/f for optimization
			float t1 = (e + in_box->hlengths[i])*m;
			float t2 = (e - in_box->hlengths[i])*m;
			// make sure that the minimum is stored in t1
			if (t1>t2)
			{
				float x=t1;
				t1=t2;t2=x; 
			}
			if (t1>tMin) {tMin=t1;hitSideMin=i;hitMinSign=-f/abs_f;} // store "side" for min hit, and the "sign" based on f.
			if (t2<tMax) {tMax=t2;hitSideMax=i;hitMaxSign=-f/abs_f;} // the sign is used to determine back or front of side (ie. if mirror of side normal)
			if (tMin>tMax) return false; // ray misses box
			if (tMax<0.001f) return false; // ray misses box
		}
		else if (-e-in_box->hlengths[i]>0 || -e+in_box->hlengths[i]<0)
		{ 
			// if the ray is parallell to the slab it can't hit it
			return false;
		}
	}

	// when all collision checking against all sides has been completed
	// successfully, save the result in hitData
	if(tMin > 0.001f && tMin < inout_intersection->dist)
	{
		if (storeResult)
		{
			inout_intersection->dist = tMin; 
			inout_intersection->surface=in_box->mat;
			inout_intersection->pos=in_ray->origin+tMin*in_ray->dir;
			inout_intersection->normal=hitMinSign*in_box->sides[hitSideMin];
		}
		return true;
	}
	else if(tMax> 0.001f && tMax < inout_intersection->dist)
	{
		if (storeResult)
		{
			inout_intersection->dist = tMax; 
			inout_intersection->surface=in_box->mat;
			inout_intersection->pos=in_ray->origin+tMax*in_ray->dir;
			inout_intersection->normal=hitMaxSign*in_box->sides[hitSideMax];
		}
		return true;
	}
	return false;
}

#endif
