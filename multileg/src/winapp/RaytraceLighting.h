#ifndef RAYTRACE_LIGHTNING_H
#define RAYTRACE_LIGHTNING_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include "KernelMathHelper.h"
#include "LightStructures.h"
#include "IntersectionInfo.h"


__device__ SurfaceLightingData* Lambert(SurfaceLightingData* inout_surfaceLight, 
										const Light* in_light, 
										const Intersection* in_intersection, 
										bool storeResult)
{
	float val=cu_dot(in_light->vec,in_intersection->normal)*in_light->diffusePower;
	inout_surfaceLight->diffuseColor=make_float4(val,val,val,0.0f);
	return inout_surfaceLight;
}

__device__ SurfaceLightingData* BlinnPhong(SurfaceLightingData* inout_surfaceLight, 
										   const Light* in_light, 
										   const float4* in_viewDir, 
										   const Intersection* in_intersection)
{
	//if (in_light->diffusePower>0.0f)
	{
		// const Material* surface = in_intersection->surface;  // it is not allowed to change the material's values from the intersection data
		// find vector between light and surface, and its length
		float4 lightSurfDir = in_light->vec - in_intersection->pos*in_light->vec.w;
		lightSurfDir *= -1.0f-in_light->vec.w;
		float sqrLightDist = 1.0f;
		if(in_light->vec.w>0.5f) sqrLightDist=squaredLen(&lightSurfDir); // if positional, distance affect
		lightSurfDir = cu_normalize(lightSurfDir);


		// diffuse light
		float ndotl = cu_dot(lightSurfDir,in_intersection->normal);
		float intensity = cu_saturate(ndotl);
		inout_surfaceLight->diffuseColor = intensity * in_light->diffuseColor * in_light->diffusePower/sqrLightDist;

		// half vector between light and ray
		float4 h = cu_normalize(lightSurfDir-*in_viewDir);

		// specular light (reuse intensity var)
		float ndoth = cu_dot(h,in_intersection->normal);

		intensity = pow(cu_fmaxf(ndoth,0.0f), in_intersection->surface.specular.w); // specular.w is glossiness value
		inout_surfaceLight->specularColor = intensity * in_light->specularColor * in_light->specularPower/sqrLightDist;
	}
	return inout_surfaceLight;
}


// __device__ SurfaceLightingData* BlinnPhongDir(SurfaceLightingData* inout_surfaceLight, 
// 											  const Light* in_light, 
// 											  const float4* in_viewDir, 
// 											  const Intersection* in_intersection)
// {
// 	//if (in_light->diffusePower>0.0f)
// 	{
// 		float4 lightSurfDir = in_light->vec;
// 
// 		// diffuse light
// 		float intensity = cu_fmaxf(cu_fminf(cu_dot(lightSurfDir,in_intersection->normal),1.0f),0.0f);
// 		inout_surfaceLight->diffuseColor = intensity*in_light->diffuseColor*in_light->diffusePower;
// 
// 		// half vector between light and ray
// 		float4 h = cu_normalize(lightSurfDir+*in_viewDir);
// 		
// 		// specular light (resuse intensity var)
// 		intensity = pow(cu_fmaxf(cu_dot(in_intersection->normal,h),0.0f), in_intersection->surface.specular.w); // specular.w is glossiness value
// 		inout_surfaceLight->specularColor = intensity*in_light->specularColor*in_light->specularPower;
// 		
// 	}
// 	return inout_surfaceLight;
// }

#endif