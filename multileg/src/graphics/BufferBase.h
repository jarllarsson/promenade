#pragma once

#include <d3d11.h>
#include "BufferConfig.h"

// =======================================================================================
//                                      BufferBase
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Base class that implement basic buffer functionality for
/// reading/writing to buffers.
///        
/// # BufferBase
/// Detailed description.....
/// Tweaked: 19-4-2013 Jarl Larsson
/// Based on BufferFactory code of Amalgamation 
///---------------------------------------------------------------------------------------

class BufferBase
{
public:
	BufferBase(ID3D11Device* p_device, ID3D11DeviceContext* p_deviceContext, 
			   BufferConfig::BUFFER_INIT_DESC& p_configDesc);
	virtual ~BufferBase();
	///-----------------------------------------------------------------------------------
	/// Apply changes made to buffer.
	/// \param misc
	/// \return void
	///-----------------------------------------------------------------------------------
	void			apply();
	void			unApply();

	ID3D11Buffer*	getBufferPointer();
	const BufferConfig*	getBufferConfigPointer();

	UINT32			getElementSize();
	UINT32			getElementCount();
protected:
	enum ApplyMode
	{
		APPLY,
		UNAPPLY,
	};
	void			init(void* p_initData);
	void*			map();
	void			unmap();
	void bufApply(ApplyMode p_mode);
	
	BufferConfig*	m_config;
private:
	ID3D11Buffer*	m_buffer;

	ID3D11Device*			m_device;
	ID3D11DeviceContext*	m_deviceContext;
};