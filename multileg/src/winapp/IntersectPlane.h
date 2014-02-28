#ifndef INTERSECT_PLANE_H
#define INTERSECT_PLANE_H

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



__device__ bool IntersectPlane(const Plane* in_plane, const Ray* in_ray, Intersection* inout_intersection, bool storeResult)
{
	// sphere intersection
	float t = (in_plane->distance - cu_dot(in_ray->origin,in_plane->normal)) / cu_dot(in_ray->dir,in_plane->normal);
	if(t > 0.001f && t < inout_intersection->dist)
	{
		if (storeResult)
		{
			inout_intersection->dist = t; 
			inout_intersection->surface=in_plane->mat;
			inout_intersection->pos=in_ray->origin+t*in_ray->dir;
			inout_intersection->normal=in_plane->normal;
		}
		return true;
	}
	return false;
}

#endif