#pragma once
#include <d3d11.h>
#include "InteropResourceMapping.h"
#include "RaytraceKernel.h"
#include <vector>
#include "TempController.h"

using namespace std;

// =======================================================================================
//                                      KernelDevice
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Manager of kernels
///        
/// # KernelDevice
/// 
/// 21-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class KernelDevice
{
public:
	enum KernelJob
	{
		J_RAYTRACEWORLD,
	};

	KernelDevice(void* p_device);
	virtual ~KernelDevice();

	void registerCanvas(void** p_texture);

	void update(float p_dt, TempController* p_tmpCam, int p_drawMode, int p_shadowMode );

	void executeKernelJob( float p_dt, KernelJob p_jobId );
protected:
private:
	int m_width, m_height;
	RaytraceKernel* m_raytracer;

	RaytraceConstantBuffer m_cb;
	InteropResourceMapping	m_gbufferHandle;
	ID3D11Device*			m_device;
};