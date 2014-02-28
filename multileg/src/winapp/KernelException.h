#pragma once
#include <BaseException.h>

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>

// =======================================================================================
//                                      KernelException
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # KernelException
/// 
/// 21-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class KernelException : public BaseException
{
public:
	KernelException(const string& p_msg,
		const string &p_file,const string &p_func,int p_line) 
		: BaseException(Stringify(KernelException),p_msg,p_file,p_func,p_line){};

	KernelException(cudaError_t p_errorCode,const string& p_msg,
		const string &p_file,const string &p_func,int p_line) 
		: BaseException(Stringify(GraphicsException),handleCUDAError(p_errorCode,p_msg),p_file,p_func,p_line){};


protected:
private:	
	string handleCUDAError(cudaError_t p_errorCode,const string& p_msg) 
	{
		string msg=" "+p_msg;
		
		if(p_errorCode!=cudaSuccess) 
		{
			msg += "("+string(cudaGetErrorString(p_errorCode))+")"; 
		}
		else
		{
			msg+="(Unknown CUDA error)";
		}

		return msg;
	}
};