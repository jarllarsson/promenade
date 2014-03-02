#include "Mesh.h"
#include "PVertex.h"

Mesh::Mesh( Buffer<PVertex>* p_vertexBuffer, 
		    Buffer<unsigned int>* p_indexBuffer )
{
	m_vertexBuffer	= p_vertexBuffer;
	m_indexBuffer	= p_indexBuffer;
}

Mesh::~Mesh()
{
	delete m_vertexBuffer;
	delete m_indexBuffer;
}

Buffer<PVertex>* Mesh::getVertexBuffer()
{
	return m_vertexBuffer;
}

Buffer<unsigned int>* Mesh::getIndexBuffer()
{
	return m_indexBuffer;
}
