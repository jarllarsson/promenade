#pragma once

#include "GraphicsDeviceFactory.h"
#include <string>
#include "ShaderVariableContainer.h"

using namespace std;

class BufferFactory;
class ComposeShader;
class MeshShader;

// =======================================================================================
//                                      ShaderFactory
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Factory for constructing shaders
///        
/// # ShaderFactory
/// 
/// Created on: 11-30-2012
///
/// Tweaked: 19-4-2013 Jarl Larsson
/// Based on ShaderFactory code of Amalgamation 
///---------------------------------------------------------------------------------------

class ShaderFactory : public GraphicsDeviceFactory
{
public:
	ShaderFactory(ID3D11Device* p_device,ID3D11DeviceContext* m_deviceContext,D3D_FEATURE_LEVEL p_featureLevel);
	virtual ~ShaderFactory();

	ComposeShader* createComposeShader( const string& p_filePath);
	MeshShader* createMeshShader( const string& p_filePath);
protected:
	///-----------------------------------------------------------------------------------
	/// A helper function that takes arguments and then compiles the given shader file 
	/// and it's given entry point. Entry point is shader stage, e.g. PixelShaderFunc
	/// \param p_sourceFile
	/// \param p_entryPoint
	/// \param p_profile
	/// \param p_blob
	/// \return void
	///-----------------------------------------------------------------------------------
	void compileShaderStage(const string&p_sourceFile, const string &p_entryPoint, 
		const string &p_profile, ID3DBlob** p_blob);

	///-----------------------------------------------------------------------------------
	/// A helper function that creates and compiles all the shader stages specified
	/// \param p_filePath
	/// \param p_vs
	/// \param p_ps
	/// \return void
	///-----------------------------------------------------------------------------------
	void createAllShaderStages( VSData* p_vs=NULL, PSData* p_ps=NULL, GSData* p_gs=NULL, 
		HSData* p_hs=NULL, DSData* p_ds=NULL);

	///-----------------------------------------------------------------------------------
	/// A helper function that creates a given sampler stage
	/// \param p_samplerState
	/// \return void
	///-----------------------------------------------------------------------------------
	void createSamplerState(ID3D11SamplerState** p_samplerState);

	void createShadowSamplerState(ID3D11SamplerState** p_samplerState);

	///-----------------------------------------------------------------------------------
	/// A helper function that creates and configures the shader from the specified input
	/// \param p_shaderInitData
	/// \param p_inputLayout
	/// \param p_vsd
	/// \param p_psd
	/// \param p_samplerState
	/// \param p_gsd
	/// \param p_hsd
	/// \param p_dsd
	/// \return void
	///-----------------------------------------------------------------------------------
	void createShaderInitData(ShaderVariableContainer* p_shaderInitData,
		ID3D11InputLayout* p_inputLayout,
		VSData* p_vsd, PSData* p_psd=NULL, 
		ID3D11SamplerState* p_samplerState=NULL,
		GSData* p_gsd=NULL, HSData* p_hsd=NULL, DSData* p_dsd=NULL);



	///-----------------------------------------------------------------------------------
	/// Generate input layout for a vertex buffer
	/// \param p_vs
	/// \param p_inputLayout
	/// \param p_maxPerVertexElements Optional. Specify if there are per-instance 
	///								  aligned elements defined after a certain number 
	///								  of per-vertex aligned elements
	/// \return void
	///-----------------------------------------------------------------------------------
	void createVertexInputLayout(VSData* p_vs, ID3D11InputLayout** p_inputLayout, int p_maxPerVertexElements=-1);

private:
	void constructInputLayout(const D3D11_INPUT_ELEMENT_DESC* p_inputDesc,
		UINT p_numberOfElements,
		VSData* p_vs, ID3D11InputLayout** p_inputLayout);

	ID3D11DeviceContext* m_deviceContext;

	string m_shaderModelVersion;
	BufferFactory* m_bufferFactory;
};