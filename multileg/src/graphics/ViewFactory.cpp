#include "ViewFactory.h"


void ViewFactory::constructDSVAndSRV( ID3D11DepthStencilView** p_outDsv, ID3D11ShaderResourceView** p_outSrv, int p_width,int p_height )
{
	HRESULT hr = S_OK;

	ID3D11Texture2D* depthStencilTexture;
	D3D11_TEXTURE2D_DESC depthStencilDesc;
	depthStencilDesc.Width = p_width;
	depthStencilDesc.Height = p_height;
	depthStencilDesc.MipLevels = 1;
	depthStencilDesc.ArraySize = 1;
	depthStencilDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	depthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
	depthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	depthStencilDesc.SampleDesc.Count = 1;
	depthStencilDesc.SampleDesc.Quality = 0;
	depthStencilDesc.CPUAccessFlags = 0;
	depthStencilDesc.MiscFlags = 0;

	HRESULT createTexHr = m_device->CreateTexture2D( &depthStencilDesc,NULL,&depthStencilTexture);
	checkHRESULT( createTexHr, __FILE__, __FUNCTION__, __LINE__ );


	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
	ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
	depthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depthStencilViewDesc.Texture2D.MipSlice = 0;

	HRESULT createDepthStencilViewHr = m_device->CreateDepthStencilView(
		depthStencilTexture, &depthStencilViewDesc, p_outDsv );
	checkHRESULT( createDepthStencilViewHr, __FILE__, __FUNCTION__, __LINE__ );

	D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc;
	ZeroMemory(&shaderResourceDesc,sizeof(shaderResourceDesc));
	shaderResourceDesc.Format = DXGI_FORMAT_R32_FLOAT;
	shaderResourceDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	shaderResourceDesc.Texture2D.MostDetailedMip = 0;
	shaderResourceDesc.Texture2D.MipLevels = 1;

	HRESULT createDepthShaderResourceView = m_device->CreateShaderResourceView(
		depthStencilTexture, &shaderResourceDesc, p_outSrv );
	checkHRESULT( createDepthShaderResourceView, __FILE__, __FUNCTION__, __LINE__ );


	depthStencilTexture->Release();
}



void ViewFactory::constructRTVAndSRV( ID3D11RenderTargetView** p_outRtv, 
																 ID3D11ShaderResourceView** p_outSrv, 
																 int p_width,int p_height,
																 DXGI_FORMAT p_format )
{
	D3D11_TEXTURE2D_DESC bufferDesc;
	ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
	bufferDesc.Width = p_width;
	bufferDesc.Height = p_height;
	bufferDesc.MipLevels = 1;
	bufferDesc.ArraySize = 1;
	bufferDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc.Format = p_format;
	bufferDesc.SampleDesc.Count = 1;
	bufferDesc.SampleDesc.Quality = 0;
	bufferDesc.CPUAccessFlags = 0;
	bufferDesc.MiscFlags = 0;

	HRESULT hr = S_OK;

	ID3D11Texture2D* bufferTexture;
	hr = m_device->CreateTexture2D( &bufferDesc, NULL, &bufferTexture );		
	checkHRESULT( hr, __FILE__, __FUNCTION__, __LINE__ );

	constructRTVAndSRVFromTexture(bufferTexture,p_outRtv,p_outSrv,p_width,p_height);

	bufferTexture->Release();
}

void ViewFactory::constructRTVAndSRVFromTexture( ID3D11Texture2D* p_texture, ID3D11RenderTargetView** p_outRtv, ID3D11ShaderResourceView** p_outSrv, int p_width,int p_height)
{
	D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
	D3D11_TEXTURE2D_DESC bufferDesc;
	p_texture->GetDesc(&bufferDesc);
	renderTargetViewDesc.Format = bufferDesc.Format;
	renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	renderTargetViewDesc.Texture2D.MipSlice = 0;

	HRESULT hr = S_OK;
	hr = m_device->CreateRenderTargetView( p_texture, &renderTargetViewDesc, p_outRtv );
	checkHRESULT( hr, __FILE__, __FUNCTION__, __LINE__ );

	D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc;
	shaderResourceDesc.Format = bufferDesc.Format;
	shaderResourceDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	shaderResourceDesc.Texture2D.MipLevels = 1;
	shaderResourceDesc.Texture2D.MostDetailedMip = 0;

	hr = m_device->CreateShaderResourceView( p_texture, &shaderResourceDesc, p_outSrv );
	checkHRESULT( hr, __FILE__, __FUNCTION__, __LINE__ );
}

void ViewFactory::constructBackbuffer( ID3D11RenderTargetView** p_outRtv, IDXGISwapChain* p_inSwapChain )
{

	if( m_device == NULL ) {
		throw GraphicsException("Device not uninitialized.",
			__FILE__,__FUNCTION__,__LINE__);
	}

	HRESULT hr = S_OK;
	ID3D11Texture2D* backBufferTexture;

	hr = p_inSwapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), 
		(LPVOID*)&backBufferTexture );

	if( FAILED(hr))
		throw GraphicsException("Failed to get backbuffer from swap chain.",
		__FILE__,__FUNCTION__,__LINE__);

	if( backBufferTexture == NULL) {
		throw GraphicsException("Failed to get backbuffer from swap chain. back buffer is NULL",
			__FILE__, __FUNCTION__,__LINE__);
	}

	hr = m_device->CreateRenderTargetView( backBufferTexture, NULL, p_outRtv );
	backBufferTexture->Release();
	if( FAILED(hr) )
		throw GraphicsException("Failed to create rendertargetview from back buffer.",
		__FILE__,__FUNCTION__,__LINE__);
}

