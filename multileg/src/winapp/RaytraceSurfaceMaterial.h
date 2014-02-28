#ifndef MATERIAL_H
#define MATERIAL_H

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


using std::vector; 

struct Material
{
	float4 diffuse;
	float4 specular; // r,g,b,glossiness
	float reflection;
	float pad[3];
};

#endif