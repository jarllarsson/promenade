#pragma once
#include <d3d11.h>
#include "D3DUtil.h"

// =======================================================================================
//                                      Texture
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Texture struct, keeps track of change
///        
/// # Texture
/// 
/// 21-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class Texture
{
public:
	Texture(ID3D11Texture2D* p_textureBuffer,int p_width, int p_height/*,
			ID3D11ShaderResourceView* p_srv,ID3D11RenderTargetView* p_rtv*/)
	{
		m_textureBuffer = p_textureBuffer;
		// m_srv = p_srv;
		// m_rtv = p_rtv;
		m_width=p_width;
		m_height=p_height;
		m_dirtyBit=true;
	}

	~Texture()
	{
		SAFE_RELEASE(m_textureBuffer);
	}

	void update(int p_width, int p_height)
	{
		m_width=p_width;
		m_height=p_height;
		m_dirtyBit=true;
	}

	bool isDirty()
	{
		return m_dirtyBit;
	}

	void unsetDirtyFlag()
	{
		m_dirtyBit=false;
	}

	ID3D11Texture2D* m_textureBuffer;
	// ID3D11ShaderResourceView* m_srv;
	// ID3D11RenderTargetView* m_rtv;
	int m_width, m_height;
private:
	bool m_dirtyBit;

};