#pragma once

#include <d3d11.h>
#include "ShaderStageData.h"
#include "ShaderVariableContainer.h"

// =======================================================================================
//                                      ShaderBase
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Shader base class containing the base functionality for every shader
///        
/// # ShaderBase
/// Detailed description.....
/// Created on: 30-11-2012 
/// Tweaked: 19-4-2013 Jarl Larsson
/// Based on ShaderFactory code of Amalgamation 
///---------------------------------------------------------------------------------------

class ShaderBase
{
public:
	ShaderBase(ShaderVariableContainer p_initData);
	virtual ~ShaderBase();
	virtual void apply()=0;
	void applyStages();
	void unApplyStages();
	void applyCustomSamplerState(ID3D11SamplerState* p_sampler, UINT p_index);
protected:
	// D3D
	ID3D11Device*			m_device;
	ID3D11DeviceContext*	m_deviceContext;

	// Shader programs
	VSData* m_vertexShader;
	GSData* m_geometryShader;
	DSData* m_domainShader;
	HSData* m_hullShader;
	PSData* m_pixelShader;

	// Input layout
	ID3D11InputLayout* m_inputLayout;

	// Sampler state
	ID3D11SamplerState* m_samplerState;
};