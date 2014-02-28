#pragma once
#include <d3d11.h>
#include <Util.h>

// =======================================================================================
//                                      D3DUtil
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Useful stuff for D3D
///        
/// # D3DUtil
/// 
/// 19-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

#define SAFE_RELEASE(x) if( x ) { (x)->Release(); (x) = NULL; }



static void setDebugName( ID3D11DeviceChild* obj,const char* name );
static void setDDebugName( ID3D11Device* obj,const char* name );
#ifdef _DEBUG
#define SETDEBUGNAME(y,x) \
	setDebugName(y,x);
#define SETDDEBUGNAME(y,x) \
	setDDebugName(y,x);
#else
#define SETDEBUGNAME(y,x)
#define SETDDEBUGNAME(y,x)
#endif
void setDebugName(ID3D11DeviceChild* obj,const char* name)
{
	obj->SetPrivateData( WKPDID_D3DDebugObjectName, (UINT)(strlen( name )), (const void*)name );
}
void setDDebugName(ID3D11Device* obj,const char* name)
{
	obj->SetPrivateData( WKPDID_D3DDebugObjectName, (UINT)(strlen( name )), (const void*)name );
}

