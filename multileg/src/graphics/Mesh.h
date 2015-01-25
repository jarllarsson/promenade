#pragma once

#include "Buffer.h"

struct Vertex;

// =======================================================================================
//                                      Mesh
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Wrapper class for buffers describing a mesh.
///        
/// # Mesh
/// Detailed description.....
/// Created: 1-3-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class Mesh
{
public:
	Mesh( Buffer<Vertex>* p_vertexBuffer, 
		  Buffer<unsigned int>* p_indexBuffer);

	///-----------------------------------------------------------------------------------
	/// The Managers for the mesh will handle the deletion of each entities data.
	/// \return 
	///-----------------------------------------------------------------------------------
	virtual ~Mesh();

	///-----------------------------------------------------------------------------------
	/// Get a pointer to the vertex buffer.
	/// \return Buffer<PNTVertex>*
	///-----------------------------------------------------------------------------------
	Buffer<Vertex>*		getVertexBuffer();

	///-----------------------------------------------------------------------------------
	/// Get a pointer to the index buffer.
	/// \return Buffer<DIndex>*
	///-----------------------------------------------------------------------------------
	Buffer<unsigned int>*	getIndexBuffer();
	unsigned int getVertexSize();
private:
	Buffer<Vertex>* m_vertexBuffer;
	Buffer<unsigned int>* m_indexBuffer;	
};