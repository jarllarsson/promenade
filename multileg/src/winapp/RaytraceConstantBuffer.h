#ifndef RAYTRACER_CONSTANTBUFFER_H
#define RAYTRACER_CONSTANTBUFFER_H

#define RAYTRACEDRAWMODE_REGULAR 0
#define RAYTRACEDRAWMODE_BLOCKDBG 1
#define RAYTRACESHADOWMODE_OFF 0
#define RAYTRACESHADOWMODE_HARD 1
#define RAYTRACESHADOWMODE_SOFT 2 // add degree of softness to this value

// =======================================================================================
//                                   RaytraceConstantBuffer
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	A constant buffer for the raytracer kernel
///        
/// # RaytraceConstantBuffer
/// 
/// 22-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

struct RaytraceConstantBuffer
{
	// TODO Split up per change rate
	int m_drawMode; // per-frame
	int m_shadowMode;  // per-frame
	float m_rayDirScaleX; // per-window change
	float m_rayDirScaleY; // per-window change
	float m_time; // per-frame
	float m_cameraRotationMat[16]; // per-frame
	float m_camPos[4]; // per-frame
};

#endif