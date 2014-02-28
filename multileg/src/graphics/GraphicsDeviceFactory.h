#pragma once

#include <d3d11.h>
#include "GraphicsException.h"
// =======================================================================================
//                                   GraphicsDeviceFactory
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Base class for factories that needs access to the device and devicecontext
///        
/// # GraphicsDeviceFactory
/// 
/// 18-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class GraphicsDeviceFactory
{
public:
	GraphicsDeviceFactory(ID3D11Device* p_device)
	{
		m_device=p_device;
	}
	virtual ~GraphicsDeviceFactory() {}
protected:
	void checkHRESULT(HRESULT p_res,const string& p_file,
		const string& p_function, int p_line)
	{
		if ( p_res != S_OK ) {
			throw GraphicsException( p_res, p_file, p_function, p_line );
		}
	}

	ID3D11Device* m_device;
private:
};