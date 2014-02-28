// =======================================================================================
//                                      BufferConfig
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Configuration struct for buffers. Defines read/write settings, size and
/// element count.
///        
/// # Buffer
/// Detailed description.....
/// Tweaked: 19-4-2013 Jarl Larsson
/// Based on BufferFactory code of Amalgamation 
///---------------------------------------------------------------------------------------
#pragma once

#include <d3d11.h>


struct BufferConfig
{
	enum BUFFER_TYPE
	{
		VERTEX_BUFFER,
		INDEX_BUFFER,
		CONSTANT_BUFFER_VS,
		CONSTANT_BUFFER_GS,
		CONSTANT_BUFFER_PS,
		CONSTANT_BUFFER_VS_PS,
		CONSTANT_BUFFER_VS_GS_PS,
		CONSTANT_BUFFER_GS_PS,
		CONSTANT_BUFFER_VS_GS,
		CONSTANT_BUFFER_ALL,
		BUFFER_TYPE_COUNT
	};

	enum BUFFER_USAGE
	{
		BUFFER_DEFAULT,
		BUFFER_STREAM_OUT_TARGET,
		BUFFER_CPU_WRITE,
		BUFFER_CPU_WRITE_DISCARD,
		BUFFER_CPU_READ,
		BUFFER_USAGE_COUNT
	};

	enum BUFFER_SLOT
	{
		PERFRAME,
		PEROBJECT,
		SLOT0,
		SLOT1,
	};
	enum VERTEX_BUFFER_SLOT{
		MISCSLOT,
		SHIPSLOT,
	};

	struct BUFFER_INIT_DESC
	{
		BUFFER_TYPE		Type;
		UINT32			NumElements;
		UINT32			ElementSize;
		BUFFER_USAGE	Usage;
		BUFFER_SLOT		Slot;
	};

	BufferConfig(BUFFER_INIT_DESC& p_initDesc);

	BUFFER_TYPE		type;
	BUFFER_USAGE	usage;

	UINT32			slot;

	UINT32			elementSize;
	UINT32			elementCount;

	const D3D11_BUFFER_DESC* getBufferDesc() const;
private:
	D3D11_BUFFER_DESC m_bufferDesc;
};