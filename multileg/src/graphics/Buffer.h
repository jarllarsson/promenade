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
		// Access buffer should only be allocated for types that has write properties
		if (m_config->usage != BufferConfig::BUFFER_DEFAULT)
		{
			unsigned int arrSz = p_configDesc.arraySize;
			// if defined as a whole array of element clusters, allocate an array
			// for memberwise access abilities as well
			// For example writeable instance buffers
			if (arrSz > 0)
			{
				accessBufferArr = new T[arrSz];
				for (unsigned int i = 0; i < arrSz; i++)
					accessBufferArr[i] = p_initData[i];
			}
			else // otherwise, store as single copy, for example constant buffers
			{
				accessBuffer = *p_initData;
			}
		}
			
		init(reinterpret_cast<void*>(p_initData));
	}
	virtual ~Buffer() 
	{
		if (m_config->arraySize > 0)
		{
			delete[] accessBufferArr;
		}
	}

	///-----------------------------------------------------------------------------------
	/// Update buffer on GPU from CPU representation (accessBuffer)
	/// \return void
	///-----------------------------------------------------------------------------------
	void update()
	{
		if (m_config->usage!=BufferConfig::BUFFER_DEFAULT) 
		{
			void* bufferGenericData = map();
			unsigned int arrSz = m_config->arraySize;
			if (arrSz > 0) // multi element copy
			{
				T* bufArr = reinterpret_cast<T*>(bufferGenericData);
				for (unsigned int i = 0; i < arrSz; i++)
					bufArr[i] = accessBufferArr[i];
			}
			else // single element copy
			{
				T* buf = reinterpret_cast<T*>(bufferGenericData);
				*buf = accessBuffer;
			}
			unmap();
		}
	}

	///
	/// Buffer for CPU access, gets copied to GPU(map/write/unmap) on update
	///
	T accessBuffer;
	T* accessBufferArr;

	// Single element access on array buffers
	T readElementAt(int p_idx);
	T* readElementPtrAt(int p_idx);
	unsigned int getArraySize();

	void writeElementAt(int p_idx, T* p_elem);

protected:
private:
};

template <class T>
unsigned int Buffer<T>::getArraySize()
{
	return m_config->arraySize;
}

template <class T>
T* Buffer<T>::readElementPtrAt(int p_idx)
{
	if (m_config->arraySize > 0)
		return &accessBufferArr[p_idx];
	else
		return &accessBuffer;
}

template <class T>
void Buffer<T>::writeElementAt(int p_idx, T* p_elem)
{
	if (m_config->arraySize > 0)
		accessBufferArr[p_idx]=*p_elem;
	else
		accessBuffer=*p_elem;
}

template <class T>
T Buffer<T>::readElementAt(int p_idx)
{
	if (m_config->arraySize > 0)
		return accessBufferArr[p_idx];
	else
		return accessBuffer;
}
