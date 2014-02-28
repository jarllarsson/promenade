#pragma once

#define R_CH 0
#define G_CH 1
#define B_CH 2
#define A_CH 3
#ifndef __CUDACC__ 
	#define __CUDACC__
#endif

// =======================================================================================
//                                    KernelMathHelper
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Various math helper functions, based on CUDA 5.0 SDK samples, 
/// with my own additions. Mainly for matrix math.
///        
/// # KernelMathHelper
/// 
/// 2013 Jarl Larsson
///---------------------------------------------------------------------------------------

#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "KernelMathOperators.h"
#include "KernelMathMatrix.h"

#pragma comment(lib, "cudart") 

inline __device__ float cu_fminf(float a, float b)
{
	return a < b ? a : b;
}
inline __device__ float cu_fmaxf(float a, float b)
{
	return a > b ? a : b;
}
inline __device__ float cu_clamp(float val, float minval, float maxval)
{
	return cu_fmaxf(cu_fminf(val,maxval),minval);
}
inline __device__ float cu_saturate(float val)
{
	return cu_clamp(val,0.0f,1.0f);
}
inline __device__ float cu_rsqrtf(float x)
{
	return 1.0f/__fsqrt_rd(x);
	//return 1.0f / sqrtf(x);
}

inline __device__ float cu_dot(const float2& a, const float2& b)
{
	return a.x * b.x + a.y * b.y;
}
inline __device__ float cu_dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __device__ float cu_dot(const float4& a, const float4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __device__ unsigned int float4ToInt(float4* rgba)
{ 
    return ((unsigned int)(rgba->w*255.0f)<<24) | 
			((unsigned int)(rgba->z*255.0f)<<16) | 
			((unsigned int)(rgba->y*255.0f)<<8) | 
			(unsigned int)(rgba->x*255.0f);
}

inline __device__ float4* mat4mul(float* mat4,const float4* in_vec, float4* out_res)
{
	out_res->x = cu_dot( *in_vec,  make_float4(mat4[0],mat4[1],mat4[2],   mat4[3]));
	out_res->y = cu_dot( *in_vec,  make_float4(mat4[4],mat4[5],mat4[6],   mat4[7]));
	out_res->z = cu_dot( *in_vec,  make_float4(mat4[8],mat4[9],mat4[10],  mat4[11]));
	out_res->w = cu_dot( *in_vec,  make_float4(mat4[12],mat4[13],mat4[14],mat4[15]));
	return out_res;
}

inline __device__ float4* mat4mul(float4x4* mat4,const float4* in_vec, float4* out_res)
{
	return mat4mul(mat4->m,in_vec, out_res);
}

// optimization for when w is not needed
inline __device__ float4* mat4mul_ignoreW(float* mat4,const float4* in_vec, float4* out_res)
{
	out_res->x = cu_dot( *in_vec,  make_float4(mat4[0],mat4[1],mat4[2],   mat4[3]));
	out_res->y = cu_dot( *in_vec,  make_float4(mat4[4],mat4[5],mat4[6],   mat4[7]));
	out_res->z = cu_dot( *in_vec,  make_float4(mat4[8],mat4[9],mat4[10],  mat4[11]));
	out_res->w = 0.0f;
	return out_res;
}

inline __device__ float4* mat4mul_ignoreW(float4x4* mat4,const float4* in_vec, float4* out_res)
{
	return mat4mul_ignoreW(mat4->m,in_vec, out_res);
}

// squared length of vector
inline __device__ float squaredLen(const float4* in_vec)
{
	return in_vec->x*in_vec->x + in_vec->y*in_vec->y + in_vec->z*in_vec->z;
}



////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cu_normalize(const float2& v)
{
	float invLen = cu_rsqrtf(cu_dot(v, v));
	return v * invLen;
}
inline __device__ float3 cu_normalize(const float3& v)
{
	float invLen = cu_rsqrtf(cu_dot(v, v));
	return v * invLen;
}
inline __device__ float4 cu_normalize(const float4& v)
{
	float invLen = cu_rsqrtf(cu_dot(v, v));
	return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __device__ float cu_length(const float2& v)
{
	//return sqrtf(cu_dot(v, v));
	return __fsqrt_rd(cu_dot(v, v));
}
inline __device__ float cu_length(const float3& v)
{
	//return sqrtf(cu_dot(v, v));
	return __fsqrt_rd(cu_dot(v, v));
}
inline __device__ float cu_length(const float4& v)
{
	//return sqrtf(cu_dot(v, v));
	return __fsqrt_rd(cu_dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////


inline __device__ float2 cu_fmodf(const float2& a, const float2& b)
{
	return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __device__ float3 cu_fmodf(const float3& a, const float3& b)
{
	return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __device__ float4 cu_fmodf(const float4& a, const float4& b)
{
	return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __device__ float3 cu_reflect(const float3& i, const float3& n)
{
	return i - 2.0f * n * cu_dot(n,i);
}

inline __device__ void cu_reflect(const float3& i, const float3& n, float3& out_reflectedVector)
{
	out_reflectedVector=i - 2.0f * n * cu_dot(n,i);
}

inline __device__ float4 cu_reflect(const float4& i, const float4& n)
{
	return i - 2.0f * n * cu_dot(n,i);
}

inline __device__ void cu_reflect(const float4& i, const float4& n, float4& out_reflectedVector)
{
	out_reflectedVector=i - 2.0f * n * cu_dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __device__ float3 cu_cross(const float3& a, const float3& b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// Extra make functions for array input
////////////////////////////////////////////////////////////////////////////////
static __inline__ __host__ __device__ float2 make_float2(float val[])
{
	float2 t; t.x = val[0]; t.y = val[1]; return t;
}

static __inline__ __host__ __device__ float3 make_float3(float val[])
{
	float3 t; t.x = val[0]; t.y = val[1]; t.z = val[2]; return t;
}

static __inline__ __host__ __device__ float4 make_float4(float val[])
{
	float4 t; t.x = val[0]; t.y = val[1]; t.z = val[2]; t.w = val[3]; return t;
}

static __inline__ __host__ __device__ float4x4 make_float4x4(float val[])
{
	float4x4 t;
	#pragma unroll
	for (int i=0;i<16;i++)
	{
		t.m[i]=val[i];
	}
	return t;
}

////////////////////////////////////////////////////////////////////////////////
// decimals
////////////////////////////////////////////////////////////////////////////////

static __inline__ __host__ __device__ float frac(float v)
{
	return v - floor(v);
}


////////////////////////////////////////////////////////////////////////////////
// Random
////////////////////////////////////////////////////////////////////////////////

// Input: It uses texture coords as the random number seed.
// Output: Random number: [0,1), that is between 0.0 and 0.999999... inclusive.
// Author: Michael Pohoreski
// Copyright: Copyleft 2012 :-)
static __device__ float cu_random( float2 p )
{
	// We need irrationals for pseudo randomness.
	// Most (all?) known transcendental numbers will (generally) work.
	const float2 r = make_float2(
		23.1406926327792690f,  // e^pi (Gelfond's constant)
		2.6651441426902251f); // 2^sqrt(2) (Gelfond–Schneider constant)
	return frac( cos( fmod( 123456789., 1e-7 + 256. *cu_dot(p,r) ) ) );  
}

static __device__ float2 cu_getRandomVector2( float2 uv )
{
	//return normalize(gRandomNormals.Sample(pointSampler, gRenderTargetSize*uv / randSize).xy * 2.0f - 1.0f);

	float2 rand;
	rand.x = cu_random( make_float2(uv.x,uv.y) );
	rand.y = cu_random( make_float2(uv.y,uv.x) );
	rand = cu_normalize( rand );
	return rand;
}

static __device__ float3 cu_getRandomVector3( float2 uv )
{
	//return normalize(gRandomNormals.Sample(pointSampler, gRenderTargetSize*uv / randSize).xy * 2.0f - 1.0f);

	float3 rand;
	rand.x = cu_random( make_float2(uv.x,uv.y) );
	rand.y = cu_random( make_float2(uv.y,uv.x) );
	rand.z = cu_random( make_float2(rand.y,uv.y*0.5f) );
	rand = cu_normalize( rand );
	return rand;
}