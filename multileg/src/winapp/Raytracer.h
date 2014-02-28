#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <iostream> 
#include <vector> 
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RaytraceConstantBuffer.h"
#include "KernelHelper.h"
#include "KernelMathHelper.h"
#include "IntersectAll.h"
#include "RaytraceShadow.h"
#include "Scene.h"
#include "Ray.h"
#include "IntersectionInfo.h"

 
#pragma comment(lib, "cudart") 

 
using std::cerr; 
using std::cout; 
using std::endl; 
using std::exception; 
using std::vector; 

// =======================================================================================
//                                   Raytracer
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	The main loop for the raytracer, here a color is determined for the pixel
///        
/// # Raytracer
/// 
/// 24-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------


__device__ __constant__ RaytraceConstantBuffer cb[1];


__device__ void Raytrace(float* p_outPixel, const int p_x, const int p_y,
						 const int p_width, const int p_height)
{
	// Normalized device coordinates of pixel. (-1 to 1)
	const float u = (p_x / (float) p_width)*2.0f-1.0f;
	const float v = (p_y / (float) p_height)*2.0f-1.0f;


	// Store contents of constant buffer in local mem
	float time = cb[0].m_time;
	float rayDirScaleX = cb[0].m_rayDirScaleX;
	float rayDirScaleY = cb[0].m_rayDirScaleY;
	int drawMode=cb[0].m_drawMode;
	int shadowMode=cb[0].m_shadowMode;
	float partLit=1.0f/(float)shadowMode;
	float4  camPos = make_float4(cb[0].m_camPos);
	float4x4 camRotation = make_float4x4(cb[0].m_cameraRotationMat);

	// =======================================================
	//                   TEST SETUP CODE
	// =======================================================
	
	// define a scene
	Scene scene;

	// define some spheres
	scene.sphere[0].pos = make_float4(0.0f,0.0f,0.0f,1.0f);
	scene.sphere[0].rad = 0.5f;
	scene.sphere[0].mat.diffuse = make_float4(0.5f, 0.79f, 0.22f,1.0f);
	scene.sphere[0].mat.specular = make_float4(1.0f, 1.0f, 1.0f,500.0f);
	scene.sphere[0].mat.reflection = 0.0f;

	scene.sphere[1].pos = make_float4(1.0f,0.0f,0.0f,1.0f);
	scene.sphere[1].rad = 0.6f;
	scene.sphere[1].mat.diffuse = make_float4(0.0f, 1.0f, 0.0f,1.0f);
	scene.sphere[1].mat.specular = make_float4(0.0f, 0.0f, 0.0f,0.0f);
	scene.sphere[1].mat.reflection = 0.0f;

	for (int i=2;i<AMOUNTOFSPHERES;i++)
	{
		scene.sphere[i].pos = make_float4((float)(i%3),(float)i,(float)i,1.0f);
		scene.sphere[i].rad = i*0.1f;
		scene.sphere[i].mat.diffuse = make_float4((float)i/(float)AMOUNTOFSPHERES, 1.0f-((float)i/(float)AMOUNTOFSPHERES), ((float)i/(float)(AMOUNTOFSPHERES*0.2f)) ,1.0f);
		scene.sphere[i].mat.specular = make_float4(1.0f, 1.0f, 1.0f,0.8f);
		scene.sphere[i].mat.reflection = (float)i/(float)AMOUNTOFSPHERES;
	}

	// define a plane

	for (int i=0;i<AMOUNTOFPLANES;i++)
	{
		scene.plane[i].distance = -5.0f;
		scene.plane[i].normal = make_float4(0.0f,1.0f,0.0f,0.0f);
		//scene.plane[i].mat.diffuse = (float4)( 71.0f/255.0f, 21.0f/255.0f, 87.0f/255.0f ,1.0f);
		scene.plane[i].mat.diffuse = make_float4( 0.1f, 0.5f, 1.0f ,1.0f);
		scene.plane[i].mat.specular = make_float4(0.1f, 0.1f, 0.1f,0.1f);
		scene.plane[i].mat.reflection = 0.0f;

	}



	// define some tris

	for (int i=0;i<AMOUNTOFTRIS;i++)
	{

		#pragma unroll 3
		for (int x=0;x<3;x++)
		{

			scene.tri[i].vertices[x] = make_float4((float)i+x*0.5f, sin(time+(float)i+x*0.01f) + ((i%2)*2-1)*(float)(x%2)*0.5f, sin((float)(x+i)*0.5f)*-3.0f,0.0f);
		}

		scene.tri[i].mat.diffuse = make_float4( 1.0f-((float)i/(float)AMOUNTOFTRIS), (float)i/(float)AMOUNTOFTRIS, 1.0f-((float)i/(float)(AMOUNTOFTRIS*0.2f)) ,1.0f);
		scene.tri[i].mat.specular = make_float4(1.0f, 1.0f, 1.0f,0.5f);
		scene.tri[i].mat.reflection = 0.6f;

	}

	// define some boxes

	for (int i=0;i<AMOUNTOFBOXES;i++)
	{
		scene.box[i].pos = make_float4(-5.0f,10+sin((float)i)*10.0f*sin(time), i*10,0.0f) + make_float4(sin((float)i)*50.0f*(1.0f+sin(time)),
			5.0f+sin(time*0.5f)*5.0f,
			cos((float)i)*50.0f*(1.0f+sin(time)),
			0.0f);
		// float4 tesst = (float4)(1.0f,0.0f,0.0f,0.0f);
		scene.box[i].sides[0] = make_float4(1.0f,0.0f,0.0f,0.0f);  // x
		scene.box[i].sides[1] = make_float4(0.0f,1.0f,0.0f,0.0f);  // y
		scene.box[i].sides[2] = make_float4(0.0f,0.0f,1.0f,0.0f);  // z
		// mat4mul(viewMatrix,&tesst, &box[i].sides[0]);
		scene.box[i].hlengths[0] = (1+i);
		scene.box[i].hlengths[1] = (1+i);
		scene.box[i].hlengths[2] = (1+i);
		scene.box[i].mat.diffuse = make_float4( (float)(i%5)*0.5f, 1.0f-sin((float)i), ((float)i/(float)(AMOUNTOFBOXES*2.0f)) ,1.0f);
		scene.box[i].mat.specular = make_float4(0.1f, 0.1f, 0.1f,0.5f);
		scene.box[i].mat.reflection = 0.2f;
	}

	// define some lights

	for (int i=0;i<AMOUNTOFLIGHTS-1;i++)
	{
		// scene.light[i].vec = (float4)(i*5.0f*sin((1.0f+i)*time),i+sin(time),100.0f*sin(time) + i*2.0f*cos((1.0f+i)*time),1.0f);
		scene.light[i].vec = make_float4(-1.0f,2.0f,sin(time)*6.0f-3.0f,1.0f);
		scene.light[i].diffusePower = 10.0f;
		scene.light[i].specularPower = 1.0f;
		scene.light[i].diffuseColor = make_float4(1.0f,1.0f,1.0f,1.0f);
		scene.light[i].specularColor = make_float4(1.0f,1.0f,1.0f,0.0f);
	}


	// Create a directional light
	scene.light[AMOUNTOFLIGHTS-1].vec = cu_normalize(make_float4(sin(time*0.1f),-1.0f,cos(time*0.1f),0.0f));
	scene.light[AMOUNTOFLIGHTS-1].diffusePower = 1.0f;
	scene.light[AMOUNTOFLIGHTS-1].specularPower = 1.0f;
	scene.light[AMOUNTOFLIGHTS-1].diffuseColor = make_float4(1.0f, 1.0f,1.0f,1.0f);
  	scene.light[AMOUNTOFLIGHTS-1].specularColor = make_float4(1.0f,1.0f,1.0f,0.0f);
	

	// 1. Create ray
	// calculate eye ray in world space
	Ray ray;
	//ray.origin = make_float4(u,v,10.0f,1.0f);
	ray.origin = camPos;

	//ray.origin = camPos;   

	float4 viewFrameDir = cu_normalize( make_float4(u*rayDirScaleX, -v*rayDirScaleY, 1.0f,0.0f) );
	//ray.dir = make_float4(0.0f,0.0f,-1.0f,0.0f);
	ray.dir = viewFrameDir;
	mat4mul(&camRotation,&viewFrameDir, &ray.dir); // transform viewFrameDir with the viewMatrix to get the world space ray

	Ray shadowRay;

	// TRANSFORM

	// 2. Declare var for final color storage
	float4 finalColor = make_float4((1.0f+ray.dir.x)*0.5f,
		(1.0f+ray.dir.z)*0.5f,
		(1.0f+ray.dir.y)*0.5f,
		0.0f);

	Intersection intersection;
	intersection.dist = MAX_INTERSECT_DIST;
	intersection.surface.diffuse = make_float4(0.0f,0.0f,0.0f,0.0f);
	intersection.surface.specular = make_float4(0.0f,0.0f,0.0f,0.0f);
	intersection.surface.reflection= 0.0f;


	// Raytrace:
	float reflectionfactor = 0.0f;
	int max_depth = 4;
	int depth = 0;

	float4 currentColor;
	SurfaceLightingData dat;
	float4 lightColor = make_float4(0.0f,0.0f,0.0f,0.0f);	

	// =======================================================

	// Main raytrace loop

	do
	{
		currentColor = make_float4(0.0f,0.0f,0.0f,0.0f);
		IntersectAll(&scene,&ray,&intersection,false);			// Do the intersection tests
		
		if (intersection.dist >= 0.0f && intersection.dist<MAX_INTERSECT_DIST)
		{

			// finalColor=intersection.color*Lambert(&light,&intersection);
			dat.diffuseColor = make_float4(0.0f,0.0f,0.0f,0.0f);
			dat.specularColor = make_float4(0.0f,0.0f,0.0f,0.0f);
			float4 ambient=make_float4(0.0f, 0.0f, 0.0f,0.0f); // ambient
			// float4 ambient=(float4)(0.0f, 0.0f, 0.0f,0.0f); // ambient

			currentColor=ambient; // ambient base add (note: on do this on current colour for ambient on shadows)

			// add all lights

			for (int i=0;i<AMOUNTOFLIGHTS;i++)
			{				

				lightColor = make_float4(0.0f,0.0f,0.0f,0.0f);		
				BlinnPhong(&dat,&(scene.light[i]),&viewFrameDir,&intersection);
				lightColor+=intersection.surface.diffuse*dat.diffuseColor;
				lightColor+=intersection.surface.specular*dat.specularColor;

				// second intersection test for shadows, return true on very first hit
				float lightHits=0;
				if (shadowMode!=RAYTRACESHADOWMODE_OFF ) // shadow
					lightHits=ShadowCastAll(&(scene.light[i]),shadowMode,
											&shadowRay,&intersection,&scene,u,v,partLit);
				else              // no shadow
					lightHits=1.0f;

				currentColor += lightColor*lightHits;
			}


			// Add color of this pixel(modified by previous pixel's reflection value) to final
			if (depth==0)
				finalColor = currentColor;
			else
				finalColor += currentColor * reflectionfactor; 


			reflectionfactor = intersection.surface.reflection;
			if (reflectionfactor>0.01f)
			{
				ray.origin = intersection.pos;
				cu_reflect(ray.dir,intersection.normal,ray.dir);
				cu_normalize(ray.dir);
 			}

			intersection.dist = MAX_INTERSECT_DIST;

			depth++;

		}
		else
			depth=max_depth;
		
	}  while (reflectionfactor>0.01f && depth<max_depth);

	// Set the color
	float dbgGridX=(float)drawMode*((float)blockIdx.x/(float)gridDim.x);
	float dbgGridY=(float)drawMode*((float)blockIdx.y/(float)gridDim.y);
	p_outPixel[R_CH] = finalColor.x + dbgGridX; // red
	p_outPixel[G_CH] = finalColor.y + dbgGridY; // green
	p_outPixel[B_CH] = finalColor.z; // blue
	p_outPixel[A_CH] = finalColor.w; // alpha
}

#endif