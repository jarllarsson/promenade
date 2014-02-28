#pragma once

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <DebugPrint.h>
#include <string>
#include <ToString.h>

using namespace std;

// =======================================================================================
//                                      KernelHelper
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # KernelHelper
/// 
/// 23-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class KernelHelper
{
public:
	static bool assertCudaResult(cudaError_t p_result)
	{
		return p_result==cudaSuccess;
	}

	static bool assertAndPrint(cudaError_t p_result,const string &p_file,const string &p_func,int p_line)
	{
		bool res = assertCudaResult(p_result);
		if (!res)
			DEBUGPRINT(( string("\n"+string(cudaGetErrorString(p_result))+" [ "+p_file+" : "+p_func+" ("+toString(p_line)+") ]\n\n").c_str() ));
		return res;
	}

};