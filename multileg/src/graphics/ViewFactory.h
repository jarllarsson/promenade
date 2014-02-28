#pragma once

#include "GraphicsDeviceFactory.h"
#include <string>
using namespace std;

// =======================================================================================
//                                      ViewFactory
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Factory for constructing various views
///        
/// # ViewFactory
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class ViewFactory : public GraphicsDeviceFactory
{
public:
	ViewFactory(ID3D11Device* p_device) : GraphicsDeviceFactory(p_device){}
	///-----------------------------------------------------------------------------------
	/// Construct Depth Stencil View and Shader Resource View
	/// \param p_outDsv
	/// \param p_outSrv
	/// \param p_width
	/// \param p_height
	/// \return void
	///-----------------------------------------------------------------------------------
	void constructDSVAndSRV( ID3D11DepthStencilView** p_outDsv, 
		ID3D11ShaderResourceView** p_outSrv, int p_width,int p_height);


	///-----------------------------------------------------------------------------------
	/// Construct Render Target View and Shader Resource View
	/// \param p_outRtv
	/// \param p_outSrv
	/// \param p_width
	/// \param p_height
	/// \param p_format
	/// \return void
	///-----------------------------------------------------------------------------------
	void constructRTVAndSRV( ID3D11RenderTargetView** p_outRtv, 
		ID3D11ShaderResourceView** p_outSrv, int p_width,int p_height,DXGI_FORMAT p_format);

	///-----------------------------------------------------------------------------------
	/// Construct Render Target View and Shader Resource View from 
	/// an already created texture.
	/// \param p_texture
	/// \param p_outRtv
	/// \param p_outSrv
	/// \param p_width
	/// \param p_height
	/// \return void
	///-----------------------------------------------------------------------------------
	void constructRTVAndSRVFromTexture( ID3D11Texture2D* p_texture,
		ID3D11RenderTargetView** p_outRtv, ID3D11ShaderResourceView** p_outSrv, 
		int p_width,int p_height);

	void constructBackbuffer(ID3D11RenderTargetView** p_outRtv,
							 IDXGISwapChain* p_inSwapChain);
protected:

private:
};