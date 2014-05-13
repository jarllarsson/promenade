#include "BufferConfig.h"
#include "GraphicsException.h"
#include "D3DUtil.h"

BufferConfig::BufferConfig(BUFFER_INIT_DESC& p_initDesc)
{
	usage = p_initDesc.Usage;
	elementSize = p_initDesc.ElementSize;
	elementCount = p_initDesc.NumElements;
	arraySize = p_initDesc.arraySize;

	switch (p_initDesc.Slot)
	{
	case PERFRAME:
	case SLOT0:
		slot = 0;
		break;
	case PEROBJECT:
	case SLOT1:
		slot = 1;
		break;
	default:
		break;
	}

	type = p_initDesc.Type;
	switch(type)
	{
	case VERTEX_BUFFER:
		{
			m_bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;

			if(p_initDesc.Usage == BUFFER_STREAM_OUT_TARGET)
				m_bufferDesc.BindFlags |= D3D11_BIND_STREAM_OUTPUT;
		}
		break;
	case INDEX_BUFFER:
		{
			m_bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		}
		break;
	case CONSTANT_BUFFER_VS:
	case CONSTANT_BUFFER_GS:
	case CONSTANT_BUFFER_PS:
	case CONSTANT_BUFFER_VS_PS:
	case CONSTANT_BUFFER_VS_GS:
	case CONSTANT_BUFFER_VS_GS_PS:	
	case CONSTANT_BUFFER_GS_PS:
	case CONSTANT_BUFFER_ALL:
		{
			m_bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		}
		break;
	default:
		throw GraphicsException("Unknown buffer type.",__FILE__,__FUNCTION__,__LINE__);
		break;
	};

	m_bufferDesc.CPUAccessFlags = 0;
	m_bufferDesc.Usage = D3D11_USAGE_DEFAULT;

	if(usage == BUFFER_CPU_READ)
	{
		m_bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		m_bufferDesc.CPUAccessFlags |= D3D11_CPU_ACCESS_READ;
	}
	else if(usage == BUFFER_CPU_WRITE)
	{
		m_bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		m_bufferDesc.CPUAccessFlags |= D3D11_CPU_ACCESS_WRITE;
	}
	else if(usage == BUFFER_CPU_WRITE_DISCARD)
	{
		m_bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		m_bufferDesc.CPUAccessFlags |= D3D11_CPU_ACCESS_WRITE;
	}

	//Desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	m_bufferDesc.MiscFlags = 0;
	m_bufferDesc.ByteWidth = p_initDesc.NumElements * p_initDesc.ElementSize;

	//set at least 16 bytes
	if(m_bufferDesc.ByteWidth < 16)
		m_bufferDesc.ByteWidth = 16;
}

const D3D11_BUFFER_DESC* BufferConfig::getBufferDesc() const
{
	return &m_bufferDesc;
}
