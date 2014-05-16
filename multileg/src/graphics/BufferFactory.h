#pragma once

#include "Buffer.h"
#include "InstanceData.h"
#include "CBuffers.h"
#include <glm\gtc\type_ptr.hpp>

struct PVertex;
class Mesh;

// =======================================================================================
//                                      BufferFactory
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Factory used to construct various buffers
///        
/// # BufferFactory
/// Detailed description.....
/// Tweaked: 19-4-2013 Jarl Larsson
/// Based on BufferFactory code of Amalgamation 
///---------------------------------------------------------------------------------------

class BufferFactory
{
public:
	BufferFactory(ID3D11Device* p_device, ID3D11DeviceContext* p_deviceContext);
	virtual ~BufferFactory();

	Buffer<PVertex>* createFullScreenQuadBuffer();
	Buffer<PVertex>* createLineBox(float p_halfSizeLength=1.0f);
	Mesh* createBoxMesh(float p_halfSizeLength=1.0f);

	///-----------------------------------------------------------------------------------
	/// Constructs a vertex buffer of a specified type T.
	/// \param p_vertices
	/// \param p_numberOfElements
	/// \return Buffer<T>*
	///-----------------------------------------------------------------------------------
	template<typename T>
	Buffer<T>* createVertexBuffer(T* p_vertices,
								  unsigned int p_numberOfElements);

	///-----------------------------------------------------------------------------------
	/// Constructs a index buffer.
	/// \param p_indices
	/// \param p_numberOfElements
	/// \return Buffer<DIndex>*
	///-----------------------------------------------------------------------------------
	Buffer<unsigned int>* createIndexBuffer(unsigned int* p_indices,
											unsigned int p_numberOfElements);

	Buffer<Mat4CBuffer>* createMat4CBuffer();
	Buffer<glm::mat4>* createMat4InstanceBuffer( void* p_instanceList, unsigned int p_numberOfElements);
protected:
private:
	ID3D11Device* m_device;
	ID3D11DeviceContext* m_deviceContext;
	UINT32 m_elementSize;
};

template<typename T>
Buffer<T>* BufferFactory::createVertexBuffer( T* p_vertices, 
											  unsigned int p_numberOfElements )
{		
	if (p_numberOfElements == 0)
		return NULL;
	Buffer<T>* vertexBuffer;

	// Create description for buffer
	BufferConfig::BUFFER_INIT_DESC vertexBufferDesc;
	vertexBufferDesc.ElementSize = sizeof(T);
	vertexBufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
	vertexBufferDesc.NumElements = p_numberOfElements ;
	vertexBufferDesc.Type = BufferConfig::VERTEX_BUFFER;

	vertexBuffer = new Buffer<T>(m_device,m_deviceContext,
		p_vertices,vertexBufferDesc);

	return vertexBuffer;
}
