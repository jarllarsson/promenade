#pragma once

#include "GraphicsDeviceFactory.h"
#include <string>
using namespace std;

class Texture;

// =======================================================================================
//                                      TextureFactory
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # TextureFactory
/// 
/// 21-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class TextureFactory : public GraphicsDeviceFactory
{
public:
	TextureFactory(ID3D11Device* p_device) : GraphicsDeviceFactory(p_device){}
	virtual ~TextureFactory() {}

	Texture* constructTexture(int p_width,int p_height, D3D11_BIND_FLAG p_bindFlags, DXGI_FORMAT p_format);

protected:
private:
};