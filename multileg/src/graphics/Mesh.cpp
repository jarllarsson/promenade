#include "Mesh.h"
#include "Vertex.h"

Mesh::Mesh( Buffer<Vertex>* p_vertexBuffer, 
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

Buffer<Vertex>* Mesh::getVertexBuffer()
{
	return m_vertexBuffer;
}

Buffer<unsigned int>* Mesh::getIndexBuffer()
{
	return m_indexBuffer;
}

unsigned int Mesh::getVertexSize()
{
	return (unsigned int)sizeof(Vertex);
}
