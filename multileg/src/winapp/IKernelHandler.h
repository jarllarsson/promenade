#pragma once

#include "KernelHelper.h"

struct KernelData
{

};

// =======================================================================================
//                                    IKernelHandler
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Base class for general kernel handling
///        
/// # IKernelHandler
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class IKernelHandler
{
public:
	IKernelHandler();
	virtual ~IKernelHandler();
	/* For OpenCL or directCompute, loading and binding would be necessary
	void LoadProgram(const string& path);
	void BindToProgram();
	*/

	///-----------------------------------------------------------------------------------
	/// Settings to be done on data on on a per kernel basis
	/// \return void
	///-----------------------------------------------------------------------------------
	virtual void SetPerKernelArgs()=0;


	///-----------------------------------------------------------------------------------
	/// Execute kernel program
	/// \param dt
	/// \return void
	///-----------------------------------------------------------------------------------
	virtual void Execute(KernelData* p_data, float p_dt)=0;


	///-----------------------------------------------------------------------------------
	/// Get the latest timing results
	/// \return double
	///-----------------------------------------------------------------------------------
	double GetLastExecTimeNS() const {return m_profilingExecTime;}
protected:

	// Debugging
	double m_profilingExecTime;

private:
};