#pragma once

#include "BufferBase.h"

// =======================================================================================
//                                      Buffer
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Template buffer class that handles mapping/unmapping correctly 
/// based on template type.
///        
/// # Buffer
/// Detailed description.....
/// Created on: 30-11-2012 
/// Tweaked: 19-4-2013 Jarl Larsson
/// Based on BufferFactory code of Amalgamation 
///---------------------------------------------------------------------------------------

template <class T>
class Buffer : public BufferBase
{
public:
	Buffer(ID3D11Device* p_device, ID3D11DeviceContext* p_deviceContext, 
		T* p_initData, BufferConfig::BUFFER_INIT_DESC& p_configDesc)
		: BufferBase(p_device, p_deviceContext, p_configDesc)
	{
		// Access buffer *HACK*, Prettier way to solve this?
		if (m_config->usage!=BufferConfig::BUFFER_DEFAULT) 
			accessBuffer = *p_initData;
		init(static_cast<void*>(p_initData));
	}
	virtual ~Buffer() {}

	///-----------------------------------------------------------------------------------
	/// Update buffer on GPU from CPU representation (accessBuffer)
	/// \return void
	///-----------------------------------------------------------------------------------
	void update()
	{
		if (m_config->usage!=BufferConfig::BUFFER_DEFAULT) 
		{
			void* bufferGenericData = map();
			T* buf = static_cast<T*>(bufferGenericData);
			*buf = accessBuffer;
			unmap();
		}
	}

	///
	/// Buffer for CPU access, gets copied to GPU(map/write/unmap) on update
	///
	T accessBuffer;

protected:
private:
};