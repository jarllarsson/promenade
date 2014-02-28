#include "TextureFactory.h"
#include "Texture.h"

Texture* TextureFactory::constructTexture( int p_width,int p_height, D3D11_BIND_FLAG p_bindFlags, DXGI_FORMAT p_format )
{
	D3D11_TEXTURE2D_DESC bufferDesc;
	ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
	bufferDesc.Width = p_width;
	bufferDesc.Height = p_height;
	bufferDesc.MipLevels = 1;
	bufferDesc.ArraySize = 1;
	bufferDesc.BindFlags =  p_bindFlags;
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
	return new Texture(bufferTexture,p_width,p_height);
}

