#ifndef INTERSECT_ALL_H
#define INTERSECT_ALL_H


#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

#include "RaytraceConstantBuffer.h"
#include "KernelMathHelper.h"
#include "Scene.h"
#include "Ray.h"
#include "IntersectionInfo.h"

#include "IntersectSphere.h"
#include "IntersectPlane.h"
#include "IntersectTriangle.h"
#include "IntersectBox.h"


// =======================================================================================
//                                      IntersectAll
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # IntersectAll
/// 
/// 25-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

__device__ bool IntersectAll(const Scene* in_scene, const Ray* in_ray, Intersection* inout_intersection, bool breakOnFirst)
{
	bool result=false;
	bool storeResults = !breakOnFirst;

   #pragma unroll AMOUNTOFSPHERES 	
	for (int i=0;i<AMOUNTOFSPHERES;i++)
	{
		result|=IntersectSphere(&(in_scene->sphere[i]), in_ray, inout_intersection, storeResults);
		if (result && breakOnFirst) 
			return true;
	}	// for each sphere	

	// #pragma unroll AMOUNTOFPLANES 
	for (int i=0;i<AMOUNTOFPLANES;i++)
	{
		result|=IntersectPlane(&(in_scene->plane[i]), in_ray, inout_intersection,storeResults);
		if (result && breakOnFirst) 
			return true;
	}	// for each plane


	#pragma unroll AMOUNTOFTRIS
	for (int i=0;i<AMOUNTOFTRIS;i++)
	{
		result|=IntersectTriangle(&(in_scene->tri[i]), in_ray, inout_intersection,storeResults);
		if (result && breakOnFirst) 
			return true;

	}	// for each tri


	#pragma unroll AMOUNTOFBOXES
	for (int i=0;i<AMOUNTOFBOXES;i++)
	{
		result|=IntersectBox(&(in_scene->box[i]), in_ray, inout_intersection,storeResults);
		if (result && breakOnFirst) 
			return true;
	}	// for each box


	/*
	// Planet field
	Sphere f;


	f.pos = make_float4(0.0f,0.0f,0.0f,1.0f);
	f.rad = 0.5f;
	f.mat.diffuse = make_float4(1.0f, 0.0f, 1.0f,1.0f);
	f.mat.specular = make_float4(1.0f, 1.0f, 1.0f,0.5f);
	f.mat.reflection = 0.0f;


	result|=MarchSphere(&f, in_ray, inout_intersection, storeResults);
	if (result && breakOnFirst) 
		return true;



	// debug for lights
	if (!breakOnFirst)
	{
		Sphere t;
		for (int i=0;i<AMOUNTOFLIGHTS-1;i++)
		{
			t.pos = in_scene->light[i].vec;
			t.rad = 0.5f;
			t.mat.diffuse = in_scene->light[i].diffuseColor;
			t.mat.specular = make_float4(1.0f, 1.0f, 1.0f,1.0f);
			t.mat.reflection = 0.0f;
			IntersectSphere(&(t), in_ray, inout_intersection,storeResults);
		}	// debug for each light
	}
	*/
	return result;
}

#endif