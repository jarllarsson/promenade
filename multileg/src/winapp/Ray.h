#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>

struct Ray
{
	float4 dir;
	float4 origin;
};