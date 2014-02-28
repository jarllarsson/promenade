#ifndef RAYTRACEKERNELENTRY_CU
#define RAYTRACEKERNELENTRY_CU

// =======================================================================================
//                                  RaytraceKernelEntry
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Entry point from host to kernel(device)
///        
/// # RaytraceKernelEntry
/// 
/// 2012->2013 Jarl Larsson
///---------------------------------------------------------------------------------------

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"

// device specific
#include "Raytracer.h"
 
#pragma comment(lib, "cudart") 

 
using std::vector; 




texture<float, 2, cudaReadModeElementType> texRef;

__global__ void RaytraceKernel(unsigned char *p_outSurface, 
							   const int p_width, const int p_height, const size_t p_pitch)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    float *pixel;

    // discard pixel if outside
    if (x >= p_width || y >= p_height) return;

    // get a pointer to the pixel at (x,y)
    pixel = (float *)(p_outSurface + y*p_pitch) + 4*x;

	Raytrace(pixel,x,y, p_width, p_height);
}
 
// Executes CUDA kernel 
extern "C" void RunRaytraceKernel(void* p_cb,unsigned char *surface,
			int width, int height, int pitch) 
{ 
	// copy to constant buffer
	cudaError_t res = cudaMemcpyToSymbol(cb, p_cb, sizeof(RaytraceConstantBuffer));
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

	// Set up dimensions
	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

	//DEBUGPRINT(( ("\n"+toString(width)+" x "+toString(height)+" @ "+toString(1000*reinterpret_cast<RaytraceConstantBuffer*>(p_cb)->b)).c_str() ));

    RaytraceKernel<<<Dg,Db>>>((unsigned char *)surface, width, height, pitch);

	res = cudaDeviceSynchronize();
	KernelHelper::assertAndPrint(res,__FILE__,__FUNCTION__,__LINE__);

} 

#endif