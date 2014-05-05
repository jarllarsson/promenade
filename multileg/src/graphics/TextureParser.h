#pragma once
#include <windows.h>
#include <d3d11.h>
#include "GraphicsException.h"
#include "FreeImageException.h"
#include "RawTexture.h"

// =======================================================================================
//                                      TextureParseer
// =======================================================================================
///---------------------------------------------------------------------------------------
/// \brief	Texture Parser wraps FreeImage and provides Texture parsing functions.
/// This namespace wraps the functionality behind FreeImage and provides a general 
/// data-to-DX11-SubResource function which can be used by others into one light weight
/// namespace. 
///        
/// # TextureParseer
/// Detailed description.....
/// Created on: 5-12-2012 
///---------------------------------------------------------------------------------------

struct FIBITMAP;

class TextureParser
{
public:
	enum TEXTURE_TYPE { RGBA, BGRA, RGB, BGR };
public:
	///-----------------------------------------------------------------------------------
	/// Called once to initialize Free Image properly
	/// \return void
	///-----------------------------------------------------------------------------------
	static void init();

	///-----------------------------------------------------------------------------------
	/// Handles the loading and creation of textures files. Supports various of types and
	/// throws exception if creation wasn't successfully.
	/// \param p_device
	/// \param p_filePath
	/// \return ID3D11ShaderResourceView*
	///-----------------------------------------------------------------------------------
	static ID3D11ShaderResourceView* loadTexture(ID3D11Device* p_device, 
		ID3D11DeviceContext* p_context, const char* p_filePath);

	static ID3D11ShaderResourceView* createTexture( ID3D11Device* p_device, 
		ID3D11DeviceContext* p_context, const byte* p_source, int p_width, int p_height, 
		int p_pitch, int p_bitLevel,
		TEXTURE_TYPE p_type );

	///-----------------------------------------------------------------------------------
	/// Handles the loading and creation of textures files. Returns raw data.
	/// \param p_filePath
	/// \return RawTexture*
	///-----------------------------------------------------------------------------------
	static RawTexture* loadTexture(const char* p_filePath);

	static RawTexture* createTexture( FIBITMAP* p_bitmap, int p_width, int p_height);

	///-----------------------------------------------------------------------------------
	/// Generates a fallback texture used when the provided path or file is not valid
	/// \return BYTE*
	///-----------------------------------------------------------------------------------
	static BYTE* generateFallbackTexture();
};
