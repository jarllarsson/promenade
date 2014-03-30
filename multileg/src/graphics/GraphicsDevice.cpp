#include "GraphicsDevice.h"
#include "GraphicsException.h"
#include "ViewFactory.h"
#include "ShaderFactory.h"
#include "BufferFactory.h"
#include "TextureFactory.h"
#include "Texture.h"
#include "D3DUtil.h"
#include "PVertex.h"
#include "Mesh.h"


#include "ComposeShader.h"
#include "MeshShader.h"


GraphicsDevice::GraphicsDevice( HWND p_hWnd, int p_width, int p_height, bool p_windowMode )
{
	m_width=p_width;
	m_height=p_height;
	m_windowMode = p_windowMode;
	m_wireframeMode=false;
	m_interopCanvasHandle=new Texture*;

	// 1. init hardware
	initSwapChain(p_hWnd);
	initHardware();

	// 2.  init factories
	m_viewFactory = new ViewFactory(m_device);
	m_shaderFactory = new ShaderFactory(m_device,m_deviceContext,m_featureLevel);
	m_bufferFactory = new BufferFactory(m_device,m_deviceContext);
	m_textureFactory = new TextureFactory(m_device);

	// 3. init views
	initBackBuffer();
	initGBufferAndDepthStencil();

	// 4. init shaders
	m_composeShader = m_shaderFactory->createComposeShader(L"../shaders/ComposeShader.hlsl");
	m_meshShader = m_shaderFactory->createMeshShader(L"../shaders/MeshShader.hlsl");

	// 5. build states
	buildBlendStates();
	m_currentBlendStateType = BlendState::DEFAULT;
	m_blendMask = 0xffffffff;
	for (int i=0;i<4;i++) m_blendFactors[i]=1;

	buildRasterizerStates();
	m_currentRasterizerStateType = RasterizerState::DEFAULT;

	// 6. Create draw-quad
	m_fullscreenQuad = m_bufferFactory->createFullScreenQuadBuffer();
	m_boxMesh = m_bufferFactory->createBoxMesh();


	fitViewport();
}

GraphicsDevice::~GraphicsDevice()
{
	m_swapChain->SetFullscreenState(false,nullptr);
	SAFE_RELEASE(m_swapChain);
	SAFE_RELEASE(m_deviceContext);
	SAFE_RELEASE(m_device);
	//
	delete m_viewFactory;
	delete m_shaderFactory;
	delete m_bufferFactory;
	delete m_textureFactory;
	delete m_interopCanvasHandle;
	//
	delete m_composeShader;
	delete m_meshShader;
	//
	delete m_fullscreenQuad;
	delete m_boxMesh;
	//
	for (unsigned int i = 0; i < m_blendStates.size(); i++){
		SAFE_RELEASE(m_blendStates[i]);
	}
	for (unsigned int i = 0; i < m_rasterizerStates.size(); i++){
		SAFE_RELEASE(m_rasterizerStates[i]);
	}
	//
	releaseGBufferAndDepthStencil();
	releaseBackBuffer();

}


void GraphicsDevice::clearRenderTargets()
{
	float clearColorRTV[4] = { 1.0f, m_width/5000.0f,  m_height/1200.0f, 1.0f };
	float clearColorBackBuffer[4] = { m_width/5000.0f, 1.0f,  m_height/1200.0f, 1.0f };

	// clear gbuffer
	unmapAllBuffers();
	unsigned int start = GBufferChannel::GBUF_DIFFUSE;
	unsigned int end = GBufferChannel::GBUF_COUNT;
	for( unsigned int i=start; i<end; i++ ) 
	{
		m_deviceContext->ClearRenderTargetView( m_gRtv[i], clearColorRTV );
	}
	// clear backbuffer
	m_deviceContext->ClearRenderTargetView(m_backBuffer,clearColorBackBuffer);
	// clear depth stencil
	m_deviceContext->ClearDepthStencilView(m_depthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
}

void GraphicsDevice::flipBackBuffer()
{
	m_swapChain->Present( 0, 0);
}

void GraphicsDevice::updateResolution( int p_width, int p_height )
{
	m_width = p_width;
	m_height = p_height;

	setRenderTarget(RenderTargetSpec::RT_NONE);
	releaseBackBuffer();
	releaseGBufferAndDepthStencil();

	HRESULT hr;
	// Resize the swap chain
	hr = m_swapChain->ResizeBuffers(0, p_width, p_height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
	if(FAILED(hr))
		throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);

	initBackBuffer();
	fitViewport();

	initGBufferAndDepthStencil();
}

void GraphicsDevice::setWindowMode( bool p_windowed )
{
	m_windowMode=p_windowed;
	HRESULT hr = S_OK;
	hr = m_swapChain->SetFullscreenState((BOOL)!p_windowed,nullptr);
	if( FAILED(hr))
		throw GraphicsException(hr,__FILE__,__FUNCTION__,__LINE__);

}

void GraphicsDevice::fitViewport()
{
	D3D11_VIEWPORT vp;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	vp.Width	= static_cast<float>(m_width);
	vp.Height	= static_cast<float>(m_height);
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	m_deviceContext->RSSetViewports(1,&vp);
}

void GraphicsDevice::setWireframeMode( bool p_wireframe )
{
	m_wireframeMode = p_wireframe;
}

void GraphicsDevice::executeRenderPass( RenderPass p_pass, 
									   Mesh* p_mesh/*=NULL*/, 
									   BufferBase* p_cbuf/*=NULL*/, 
									   BufferBase* p_instances/*=NULL */ )
{
	switch(p_pass)
	{	
	case RenderPass::P_BASEPASS:
		if (p_mesh && p_instances)
		{
			m_deviceContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
			setBlendState(BlendState::NORMAL);
			setRasterizerStateSettings(RasterizerState::DEFAULT);
			setRenderTarget(RT_MRT);
			setShader(SI_MESHSHADER);
			drawInstancedMesh(p_mesh,p_instances);
		}
		break;
	case RenderPass::P_COMPOSEPASS:
		m_deviceContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
		setBlendState(BlendState::NORMAL);
		setRasterizerStateSettings(RasterizerState::DEFAULT,false);
		setRenderTarget(RT_BACKBUFFER_NODEPTHSTENCIL);
		setShader(SI_COMPOSESHADER);
		drawFullscreen();
		break;
	case RenderPass::P_WIREFRAMEPASS:
		if (p_instances && p_cbuf)
		{
			m_deviceContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
			setBlendState(BlendState::NORMAL);
			setRasterizerStateSettings(RasterizerState::FILLED_NOCULL,false);
			setRenderTarget(RT_BACKBUFFER_NODEPTHSTENCIL);		
			p_cbuf->apply();
			setShader(SI_WIREFRAMESHADER);
			drawInstancedAABB(p_instances);
		}
		break;
	}
}


void* GraphicsDevice::getDevicePointer()
{
	return (void*)m_device;
}





void GraphicsDevice::mapGBuffer()
{
	unsigned int start = GBufferChannel::GBUF_DIFFUSE;
	unsigned int end = GBufferChannel::GBUF_COUNT;
	for( unsigned int i=start; i<end; i++ ) 
	{
		m_deviceContext->PSSetShaderResources( i, 1, &m_gSrv[i] );
	}
}

void GraphicsDevice::mapGBufferSlot( GBufferChannel p_slot )
{
	unsigned int i = static_cast<unsigned int>(p_slot);
	m_deviceContext->PSSetShaderResources( i, 1, &m_gSrv[i] );
}

void GraphicsDevice::mapDepth()
{
	unsigned int i = static_cast<unsigned int>(GBufferChannel::GBUF_DEPTH);
	m_deviceContext->PSSetShaderResources( i, 1, &m_depthSrv );
}


void GraphicsDevice::unmapGBuffer()
{
	ID3D11ShaderResourceView* nulz = NULL;
	unsigned int start = GBufferChannel::GBUF_DIFFUSE;
	unsigned int end = GBufferChannel::GBUF_COUNT;
	for( unsigned int i=start; i<end; i++ ) 
	{
		m_deviceContext->PSSetShaderResources( i, 1, &nulz );
	}
}


void GraphicsDevice::unmapGBufferSlot( GBufferChannel p_slot )
{
	ID3D11ShaderResourceView* nulz = NULL;
	m_deviceContext->PSSetShaderResources( static_cast<unsigned int>(p_slot), 1, &nulz );
}

void GraphicsDevice::unmapDepth()
{
	ID3D11ShaderResourceView* nulz = NULL;
	m_deviceContext->PSSetShaderResources( static_cast<unsigned int>(GBufferChannel::GBUF_DEPTH), 
										   1, &nulz );
}

void GraphicsDevice::unmapAllBuffers()
{
	unmapGBuffer();
	unmapDepth();
}


void GraphicsDevice::setRenderTarget( RenderTargetSpec p_target )
{
	switch(p_target)
	{	
	case RenderTargetSpec::RT_NONE:
		m_deviceContext->OMSetRenderTargets(0,0,0);
		break;
	case RenderTargetSpec::RT_BACKBUFFER:	
		m_deviceContext->OMSetRenderTargets(1,&m_backBuffer,m_depthStencilView);
		break;
	case RenderTargetSpec::RT_BACKBUFFER_NODEPTHSTENCIL:
		m_deviceContext->OMSetRenderTargets(1,&m_backBuffer,0);
		break;
	case RenderTargetSpec::RT_MRT:
		m_deviceContext->OMSetRenderTargets(GBufferChannel::GBUF_COUNT,m_gRtv,m_depthStencilView);
		break;
	case RenderTargetSpec::RT_MRT_NODEPTHSTENCIL:
		m_deviceContext->OMSetRenderTargets(GBufferChannel::GBUF_COUNT,m_gRtv,0);
		break;
	}
}


void GraphicsDevice::setShader( ShaderId p_shaderId )
{
	switch(p_shaderId)
	{	
	case ShaderId::SI_NONE:
		m_deviceContext->PSSetShaderResources(0,0,0);
		break;
	case ShaderId::SI_MESHSHADER:
		// map resources here
		m_meshShader->apply();
	case ShaderId::SI_COMPOSESHADER:	
		mapGBuffer();
		m_composeShader->apply();
		break;
	case ShaderId::SI_WIREFRAMESHADER:	
		m_wireframeShader->apply();
		break;
	}
}

void GraphicsDevice::setBlendState(BlendState::Mode p_state)
{
	unsigned int idx = static_cast<unsigned int>(p_state);
	m_deviceContext->OMSetBlendState( m_blendStates[idx], m_blendFactors, m_blendMask );
	m_currentBlendStateType = p_state;
}

void GraphicsDevice::setBlendFactors( float p_red, float p_green, float p_blue, 
									   float p_alpha )
{
	m_blendFactors[0]=p_red;
	m_blendFactors[1]=p_green;
	m_blendFactors[2]=p_blue;
	m_blendFactors[3]=p_alpha;
}

void GraphicsDevice::setBlendFactors( float p_oneValue )
{
	for (int i=0;i<4;i++)
		m_blendFactors[i]=p_oneValue;
}

void GraphicsDevice::setBlendMask( UINT p_mask )
{
	m_blendMask = p_mask;
}

BlendState::Mode GraphicsDevice::getCurrentBlendStateType() 
{
	return m_currentBlendStateType;
}

void GraphicsDevice::setRasterizerStateSettings( RasterizerState::Mode p_state,
												bool p_allowWireframOverride/*=true*/)
{
	RasterizerState::Mode state = getCurrentRasterizerStateType();
	RasterizerState::Mode newState = p_state;
	unsigned int idx = static_cast<unsigned int>(newState);
	bool set=false;
	// accept rasterizer state change if not in wireframe mode or 
	// if set to not allow wireframe mode
	if (!m_wireframeMode || !p_allowWireframOverride)
	{	
		m_deviceContext->RSSetState( m_rasterizerStates[idx] );
		set=true;
	}
	else if (state != RasterizerState::WIREFRAME) 
	{   
		// otherwise, force wireframe(if not already set)
		idx = static_cast<unsigned int>(RasterizerState::WIREFRAME);
		set=true;
	}
	if (set)
	{
		m_deviceContext->RSSetState( m_rasterizerStates[idx] );
		m_currentRasterizerStateType = newState;
	}
}

RasterizerState::Mode GraphicsDevice::getCurrentRasterizerStateType()
{
	return m_currentRasterizerStateType;
}


void GraphicsDevice::drawFullscreen()
{
	m_fullscreenQuad->apply();
	m_deviceContext->Draw(6,0);
}


void GraphicsDevice::drawInstancedMesh( Mesh* p_mesh, BufferBase* p_instanceRef )
{
	UINT vertsz=p_mesh->getVertexBuffer()->getElementSize();
	UINT instancesz=p_instanceRef->getElementSize();
	UINT strides[2] = { vertsz, instancesz };
	UINT offsets[2] = { 0, 0 };
	// Set up an array of the buffers for the vertices
	ID3D11Buffer* buffers[2] = { 
		p_mesh->getVertexBuffer()->getBufferPointer(), 
		p_instanceRef->getBufferPointer(), 
	};
	// Set array of buffers to context 
	m_deviceContext->IASetVertexBuffers(0, 2, buffers, strides, offsets);

	// And the index buffer
	m_deviceContext->IASetIndexBuffer(p_mesh->getIndexBuffer()->getBufferPointer(), 
		DXGI_FORMAT_R32_UINT, 0);


	// Draw instanced data
	UINT32 indexCount=p_mesh->getIndexBuffer()->getElementCount();
	UINT32 instanceCount=p_instanceRef->getElementCount();
	m_deviceContext->DrawIndexedInstanced( indexCount, instanceCount, 0,0,0);
}

void GraphicsDevice::drawInstancedAABB( BufferBase* p_instanceRef )
{
	drawInstancedMesh(m_boxMesh,p_instanceRef);
}


void GraphicsDevice::initSwapChain( HWND p_hWnd )
{
	ZeroMemory( &m_swapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC) );
	m_swapChainDesc.BufferCount = 1;
	m_swapChainDesc.BufferDesc.Width = m_width;
	m_swapChainDesc.BufferDesc.Height = m_height;
	m_swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	m_swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
	m_swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	m_swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	m_swapChainDesc.OutputWindow = p_hWnd;
	m_swapChainDesc.SampleDesc.Count = 1;
	m_swapChainDesc.SampleDesc.Quality = 0;
	m_swapChainDesc.Windowed = m_windowMode;
}

void GraphicsDevice::initHardware()
{
	HRESULT hr = S_OK;
	UINT createDeviceFlags = 0;
#ifdef _DEBUG
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
	D3D_DRIVER_TYPE driverTypes[] = 
	{
		D3D_DRIVER_TYPE_HARDWARE,
		D3D_DRIVER_TYPE_REFERENCE,
	};
	UINT numDriverTypes = sizeof(driverTypes) / sizeof(driverTypes[0]);

	D3D_FEATURE_LEVEL featureLevelsToTry[] = {
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0
	};
	D3D_FEATURE_LEVEL initiatedFeatureLevel;

	int selectedDriverType = -1;

	for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
	{
		D3D_DRIVER_TYPE driverType;
		driverType = driverTypes[driverTypeIndex];
		hr = D3D11CreateDeviceAndSwapChain(
			NULL,
			driverType,
			NULL,
			createDeviceFlags,
			featureLevelsToTry,
			ARRAYSIZE(featureLevelsToTry),
			D3D11_SDK_VERSION,
			&m_swapChainDesc,
			&m_swapChain,
			&m_device,
			&initiatedFeatureLevel,
			&m_deviceContext);

		if (hr == S_OK)
		{
			selectedDriverType = driverTypeIndex;
			m_featureLevel = m_device->GetFeatureLevel();
			SETDDEBUGNAME((m_device),("m_device"));
			break;
		}
	}
	if ( selectedDriverType > 0 )
		throw GraphicsException("Couldn't create a D3D Hardware-device, software render enabled."
		,__FILE__, __FUNCTION__, __LINE__);
}

void GraphicsDevice::initBackBuffer()
{
	m_viewFactory->constructBackbuffer(&m_backBuffer,m_swapChain);
}

void GraphicsDevice::initDepthStencil()
{
	m_viewFactory->constructDSVAndSRV(&m_depthStencilView,
									  &m_depthSrv,
									  m_width,m_height);
}

void GraphicsDevice::initGBuffer()
{
	// Init all slots in gbuffer
	unsigned int start = GBufferChannel::GBUF_DIFFUSE;
	unsigned int end = GBufferChannel::GBUF_COUNT;
	for( unsigned int i=start; i<end; i++ ) 
	{
		m_gTexture[i] = m_textureFactory->constructTexture(m_width, m_height,
			(D3D11_BIND_FLAG)(/*D3D11_BIND_UNORDERED_ACCESS | */D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE),
			DXGI_FORMAT_R32G32B32A32_FLOAT); // change to DXGI_FORMAT_R8G8B8A8_UNORM or maybe 16?

		m_viewFactory->constructRTVAndSRVFromTexture( m_gTexture[i]->m_textureBuffer,
													  &m_gRtv[i], 
													  &m_gSrv[i], 
													  m_width, m_height);
	}
	*m_interopCanvasHandle = (Texture*)m_gTexture[GBufferChannel::GBUF_DIFFUSE];
}


void GraphicsDevice::initGBufferAndDepthStencil()
{
	initDepthStencil();
	initGBuffer();
}

void GraphicsDevice::buildBlendStates()
{
	RenderStateHelper::fillBlendStateList(m_device,m_blendStates);
}

void GraphicsDevice::buildRasterizerStates()
{
	RenderStateHelper::fillRasterizerStateList(m_device,m_rasterizerStates);
}

void GraphicsDevice::releaseBackBuffer()
{
	SAFE_RELEASE( m_backBuffer );
}

void GraphicsDevice::releaseGBufferAndDepthStencil()
{
	SAFE_RELEASE(m_depthStencilView);

	for (int i = 0; i < GBufferChannel::GBUF_COUNT; i++)
	{
		SAFE_DELETE(m_gTexture[i]);
		SAFE_RELEASE(m_gRtv[i]);
		SAFE_RELEASE(m_gSrv[i]);
	}
}

float GraphicsDevice::getAspectRatio()
{
	return (float)m_width/(float)m_height;
}

int GraphicsDevice::getWidth()
{
	return m_width;
}

int GraphicsDevice::getHeight()
{
	return m_height;
}
