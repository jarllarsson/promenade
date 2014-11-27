#include "ShaderFactory.h"
#include "BufferFactory.h"
#include "GraphicsException.h"
#include "ComposeShader.h"
#include "MeshShader.h"
#include <D3DCompiler.h>
#include <comdef.h>
#include <d3d11.h>
#include <vector>
#include <d3d11shader.h>
#include <DebugPrint.h>
#include <StrTools.h>


ShaderFactory::ShaderFactory(ID3D11Device* p_device, ID3D11DeviceContext* p_deviceContext, 
							 D3D_FEATURE_LEVEL p_featureLevel) : GraphicsDeviceFactory(p_device)
{
	m_device = p_device;
	m_deviceContext = p_deviceContext;
	switch(m_device->GetFeatureLevel())
	{
	case D3D_FEATURE_LEVEL_10_1:
	case D3D_FEATURE_LEVEL_10_0:
		m_shaderModelVersion = "4_0"; break;
	default:
		m_shaderModelVersion = "5_0"; break;
	}
	m_bufferFactory = new BufferFactory(m_device,m_deviceContext);

	switch( p_featureLevel )
	{
	case D3D_FEATURE_LEVEL_10_0:
		m_shaderModelVersion = "4_0";
		break;

	case D3D_FEATURE_LEVEL_10_1:
		m_shaderModelVersion = "4_0";
		break;

	case D3D_FEATURE_LEVEL_11_0:
		m_shaderModelVersion = "5_0";
		break;

	case D3D_FEATURE_LEVEL_11_1:
		m_shaderModelVersion = "5_0";
		break;
	}
}


ShaderFactory::~ShaderFactory()
{
	delete m_bufferFactory;
}


ComposeShader* ShaderFactory::createComposeShader( const string& p_filePath )
{
	ID3D11SamplerState*		samplerState = NULL;
	ID3D11InputLayout*		inputLayout = NULL;
	ShaderVariableContainer shaderVariables;

	VSData* vertexData	= new VSData();
	PSData* pixelData	= new PSData();
	
	vertexData->stageConfig = new ShaderStageConfig(p_filePath, "VS", m_shaderModelVersion);
	pixelData->stageConfig = new ShaderStageConfig(p_filePath, "PS", m_shaderModelVersion);

	createAllShaderStages(vertexData, pixelData);
	createSamplerState(&samplerState);
	createVertexInputLayout(vertexData,&inputLayout);
	createShaderInitData(&shaderVariables,inputLayout,vertexData,pixelData,samplerState,NULL);

	return new ComposeShader(shaderVariables);
}

MeshShader* ShaderFactory::createMeshShader( const string& p_filePath )
{
	ID3D11SamplerState*		samplerState = NULL;
	ID3D11InputLayout*		inputLayout = NULL;
	ShaderVariableContainer shaderVariables;

	VSData* vertexData	= new VSData();
	PSData* pixelData	= new PSData();

	vertexData->stageConfig = new ShaderStageConfig(p_filePath,"VS",m_shaderModelVersion);
	pixelData->stageConfig = new ShaderStageConfig(p_filePath,"PS", m_shaderModelVersion);

	createAllShaderStages(vertexData, pixelData);
	createSamplerState(&samplerState);
	createVertexInputLayout(vertexData,&inputLayout);
	createShaderInitData(&shaderVariables,inputLayout,vertexData,pixelData,samplerState,NULL);

	return new MeshShader(shaderVariables);
}

void ShaderFactory::compileShaderStage( const string& p_sourceFile, 
									    const string &p_entryPoint, 
										const string &p_profile, ID3DBlob** p_blob )
{
	HRESULT res = S_OK;

	ID3DBlob*	blobError  = NULL;
	ID3DBlob*	shaderBlob = NULL;

	*p_blob = NULL;

	DWORD compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;

#if defined(DEBUG) || defined(_DEBUG)
	compileFlags |= D3DCOMPILE_DEBUG; 
	compileFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
	//compileFlags |= D3DCOMPILE_WARNINGS_ARE_ERRORS;
#else
	compileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3; 
#endif

	// Compile the programs
	// vertex
	std::wstring stpath = stringToWstring(p_sourceFile);
	LPCWSTR lpath = stpath.c_str();
	res = D3DCompileFromFile(lpath, 0,
		D3D_COMPILE_STANDARD_FILE_INCLUDE,
		(LPCTSTR)p_entryPoint.c_str(), (LPCTSTR)p_profile.c_str(), 
		compileFlags, 0, 
		&shaderBlob, &blobError);
	if ( FAILED(res) )
	{
		DEBUGWARNING(((string("Error in shader:\n") + p_sourceFile+"\n(OK to read error msg)").c_str()));
		if (blobError!=NULL)
			throw GraphicsException(blobError,__FILE__,__FUNCTION__,__LINE__);
		else
			throw GraphicsException(res,__FILE__,__FUNCTION__,__LINE__);	
		return;
	}

	*p_blob = shaderBlob;
}

void ShaderFactory::createAllShaderStages(VSData* p_vs/* =NULL */, 
										  PSData* p_ps/* =NULL */, 
										  GSData* p_gs/* =NULL */, 
										  HSData* p_hs/* =NULL */, 
										  DSData* p_ds/* =NULL */)
{
	bool pixelCompiled	= false;
	bool vertexCompiled = false;
	bool geometryCompiled = false;
	bool hullCompiled	= false;
	bool domainCompiled = false;

	if (p_vs)
	{
		HRESULT hr = S_OK;
		compileShaderStage(p_vs->stageConfig->filePath,p_vs->stageConfig->entryPoint,
			string("vs_")+p_vs->stageConfig->version,&p_vs->compiledData);

		hr = m_device->CreateVertexShader(p_vs->compiledData->GetBufferPointer(),
			p_vs->compiledData->GetBufferSize(), NULL, &p_vs->data);
		if(FAILED(hr))
			throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);
		else
			vertexCompiled = true;
	}
	else
		throw GraphicsException("Missing vertex shader", __FILE__,__FUNCTION__,__LINE__);

	if(p_ps)
	{
		HRESULT hr = S_OK;
		compileShaderStage(p_ps->stageConfig->filePath,p_ps->stageConfig->entryPoint,
			string("ps_")+p_ps->stageConfig->version,&p_ps->compiledData);

		hr = m_device->CreatePixelShader(p_ps->compiledData->GetBufferPointer(),
			p_ps->compiledData->GetBufferSize(), NULL, &p_ps->data);
		if(FAILED(hr))
			throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);
		else
			pixelCompiled = true;
	}

	if(vertexCompiled)
	{
		if (p_gs)
		{
			HRESULT hr = S_OK;
			compileShaderStage(p_gs->stageConfig->filePath,p_gs->stageConfig->entryPoint,
				string("gs_")+p_gs->stageConfig->version,&p_gs->compiledData);

			hr = m_device->CreateGeometryShader(p_gs->compiledData->GetBufferPointer(),
				p_gs->compiledData->GetBufferSize(), NULL, &p_gs->data);
			if(FAILED(hr))
				throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);
			else
				geometryCompiled = true;
		}

		if (p_hs)
		{
			HRESULT hr = S_OK;
			compileShaderStage(p_hs->stageConfig->filePath,p_hs->stageConfig->entryPoint,
				string("hs_")+p_hs->stageConfig->version,&p_hs->compiledData);

			hr = m_device->CreateHullShader(p_hs->compiledData->GetBufferPointer(),
				p_hs->compiledData->GetBufferSize(), NULL, &p_hs->data);
			if(FAILED(hr))
				throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);
			else
				hullCompiled = true;
		}
	
		if(p_ds && p_hs)
		{
			HRESULT hr = S_OK;
			compileShaderStage(p_ds->stageConfig->filePath,p_ds->stageConfig->entryPoint,
				string("ds_")+p_ds->stageConfig->version,&p_ds->compiledData);

			hr = m_device->CreateDomainShader(p_ds->compiledData->GetBufferPointer(),
				p_ds->compiledData->GetBufferSize(), NULL, &p_ds->data);
			if(FAILED(hr))
				throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);
			else 
				domainCompiled = true;
		}
		else if(hullCompiled)
			throw GraphicsException("Invalid shader stage config",__FILE__,__FUNCTION__,
			__LINE__);
	}
}

void ShaderFactory::createSamplerState( ID3D11SamplerState** p_samplerState )
{
	HRESULT hr = S_OK;

	D3D11_SAMPLER_DESC samplerDesc;
	ZeroMemory(&samplerDesc,sizeof(samplerDesc));
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	//samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc.Filter = D3D11_FILTER_ANISOTROPIC; // This looks better but is more expensive /ML
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 16;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	samplerDesc.MinLOD = 0;
	samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

	hr = m_device->CreateSamplerState(&samplerDesc,p_samplerState );
	if(FAILED(hr))
		throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);
}


void ShaderFactory::createShadowSamplerState( ID3D11SamplerState** p_samplerState )
{
	HRESULT hr = S_OK;

	D3D11_SAMPLER_DESC samplerDesc;
	ZeroMemory(&samplerDesc,sizeof(samplerDesc));
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 16;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	samplerDesc.MinLOD = 0;
	samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

	hr = m_device->CreateSamplerState(&samplerDesc,p_samplerState );
	if(FAILED(hr))
		throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);
}


void ShaderFactory::createShaderInitData(ShaderVariableContainer* p_shaderInitData, 
										 ID3D11InputLayout* p_inputLayout,
										 VSData* p_vsd, PSData* p_psd/* =NULL */,
										 ID3D11SamplerState* p_samplerState/* =NULL */,
										 GSData* p_gsd/* =NULL */, 
										 HSData* p_hsd/* =NULL */, 
										 DSData* p_dsd/* =NULL */)
{
	p_shaderInitData->deviceContext = m_deviceContext;
	p_shaderInitData->inputLayout	= p_inputLayout;
	p_shaderInitData->vertexShader	= p_vsd;
	p_shaderInitData->hullShader	= p_hsd;
	p_shaderInitData->domainShader	= p_dsd;
	p_shaderInitData->geometryShader = p_gsd;
	p_shaderInitData->pixelShader	= p_psd;
	p_shaderInitData->samplerState = p_samplerState;
}



// Automatic input layout generation using shader reflection based on the article:
// http://takinginitiative.net/2011/12/11/directx-1011-basic-shader-reflection-automatic-input-layout-creation/
// Tweaked: 19-4-2013 Jarl Larsson
// Removed byteoffset calculation, not necessary to compute
// Added functionality for  

void ShaderFactory::createVertexInputLayout( VSData* p_vs, ID3D11InputLayout** p_inputLayout, int p_maxPerVertexElements/*=-1*/ )
{
	// Reflect shader info
	ID3D11ShaderReflection* vertexShaderReflection = NULL; 
	HRESULT hr = D3DReflect( p_vs->compiledData->GetBufferPointer(), 
							 p_vs->compiledData->GetBufferSize(), 
							 IID_ID3D11ShaderReflection, 
							 (void**) &vertexShaderReflection );
	if ( FAILED(hr) )
		throw GraphicsException(hr, __FILE__, __FUNCTION__, __LINE__);

	// Get shader description from reflection
	D3D11_SHADER_DESC shaderDesc;
	vertexShaderReflection->GetDesc( &shaderDesc );

	// Read input layout description from shader info
	vector<D3D11_INPUT_ELEMENT_DESC> inputLayoutDesc;
	for ( unsigned int i=0; i< shaderDesc.InputParameters; i++ )
	{
		D3D11_SIGNATURE_PARAMETER_DESC paramDesc;       
		vertexShaderReflection->GetInputParameterDesc(i, &paramDesc );

		// fill out input element desc
		D3D11_INPUT_ELEMENT_DESC elementDesc;   
		elementDesc.SemanticName = paramDesc.SemanticName;      
		elementDesc.SemanticIndex = paramDesc.SemanticIndex;
		elementDesc.AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
		int inputslot=0,instancestep=0;
		if (p_maxPerVertexElements==-1 || i<(unsigned int)p_maxPerVertexElements)
		{
			string semname(elementDesc.SemanticName);
			if (semname.substr(0,8)=="INSTANCE")
			{
				elementDesc.InputSlotClass = D3D11_INPUT_PER_INSTANCE_DATA;
				inputslot=1;
				instancestep=1;
			}
			else
				elementDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		}
		else
		{
			elementDesc.InputSlotClass = D3D11_INPUT_PER_INSTANCE_DATA;
			inputslot=1;
			instancestep=1;
		}
		elementDesc.InputSlot = inputslot;
		elementDesc.InstanceDataStepRate = instancestep;   

		// determine DXGI format
		if ( paramDesc.Mask == 1 )
		{
			if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32 ) elementDesc.Format = DXGI_FORMAT_R32_UINT;
			else if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32 ) elementDesc.Format = DXGI_FORMAT_R32_SINT;
			else if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 ) elementDesc.Format = DXGI_FORMAT_R32_FLOAT;
		}
		else if ( paramDesc.Mask <= 3 )
		{
			if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32 ) elementDesc.Format = DXGI_FORMAT_R32G32_UINT;
			else if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32 ) elementDesc.Format = DXGI_FORMAT_R32G32_SINT;
			else if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 ) elementDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
		}
		else if ( paramDesc.Mask <= 7 )
		{
			if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32 ) elementDesc.Format = DXGI_FORMAT_R32G32B32_UINT;
			else if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32 ) elementDesc.Format = DXGI_FORMAT_R32G32B32_SINT;
			else if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 ) elementDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		}
		else if ( paramDesc.Mask <= 15 )
		{
			if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_UINT32 ) elementDesc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
			else if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_SINT32 ) elementDesc.Format = DXGI_FORMAT_R32G32B32A32_SINT;
			else if ( paramDesc.ComponentType == D3D_REGISTER_COMPONENT_FLOAT32 ) elementDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		}

		//save element desc
		inputLayoutDesc.push_back(elementDesc);
	}       

	constructInputLayout(&inputLayoutDesc[0],(UINT)inputLayoutDesc.size(),p_vs,p_inputLayout);

	//Free allocation shader reflection memory
	vertexShaderReflection->Release();
}

void ShaderFactory::constructInputLayout(const D3D11_INPUT_ELEMENT_DESC* p_inputDesc, 
										 UINT p_numberOfElements,
										 VSData* p_vs, ID3D11InputLayout** p_inputLayout )
{
	HRESULT hr = m_device->CreateInputLayout(
		p_inputDesc, 
		p_numberOfElements, 
		p_vs->compiledData->GetBufferPointer(),
		p_vs->compiledData->GetBufferSize(),
		p_inputLayout);

	if ( FAILED(hr) )
		throw GraphicsException(hr, __FILE__, __FUNCTION__, __LINE__);
}