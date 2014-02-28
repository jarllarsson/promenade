// =======================================================================================
//                                      ShaderStageData
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Contains structs need for different stages of the shader program
///        
/// # ShaderStageData
/// Detailed description.....
/// Created on: 30-11-2012 
/// Tweaked: 19-4-2013 Jarl Larsson
/// Based on ShaderFactory code of Amalgamation 
///---------------------------------------------------------------------------------------
#pragma once

#include <d3d11.h>
#include "D3DUtil.h"
#include "ShaderStageConfig.h"

struct VSData
{
	ShaderStageConfig*	stageConfig;
	ID3DBlob*			compiledData;
	ID3D11VertexShader* data;
	VSData()
	{
		compiledData = NULL;
		data = NULL;
		stageConfig = NULL;
	}
	~VSData()
	{
		SAFE_RELEASE(compiledData);
		SAFE_RELEASE(data);
		delete stageConfig;
	}
};

struct GSData
{
	ShaderStageConfig*		stageConfig;
	ID3DBlob*				compiledData;
	ID3D11GeometryShader*	data;

	GSData()
	{
		compiledData = NULL;
		data = NULL;
		stageConfig = NULL;
	}

	~GSData()
	{
		SAFE_RELEASE(compiledData);
		SAFE_RELEASE(data);
		delete stageConfig;
	}
};

struct DSData
{
	ShaderStageConfig*	stageConfig;
	ID3DBlob*			compiledData;
	ID3D11DomainShader* data;

	DSData()
	{
		compiledData = NULL;
		data = NULL;
		stageConfig = NULL;
	}

	~DSData()
	{
		SAFE_RELEASE(compiledData);
		SAFE_RELEASE(data);
		delete stageConfig;
	}
};
struct HSData
{
	ShaderStageConfig*	stageConfig;
	ID3DBlob*			compiledData;
	ID3D11HullShader*	data;

	HSData()
	{
		compiledData = NULL;
		data = NULL;
		stageConfig = NULL;
	}

	~HSData()
	{
		SAFE_RELEASE(compiledData);
		SAFE_RELEASE(data);
		delete stageConfig;
	}
};
struct PSData
{
	ShaderStageConfig*	stageConfig;
	ID3DBlob*			compiledData;
	ID3D11PixelShader*	data;

	PSData()
	{
		compiledData = NULL;
		data = NULL;
		stageConfig = NULL;
	}
	~PSData()
	{
		SAFE_RELEASE(compiledData);
		SAFE_RELEASE(data);
		delete stageConfig;
	}
};