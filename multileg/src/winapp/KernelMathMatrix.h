#pragma once

#define __CUDACC__

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "KernelMathOperators.h"

#pragma comment(lib, "cudart") 

struct float4x4
{
	float m[16];
};