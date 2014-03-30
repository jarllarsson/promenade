#include "BufferFactory.h"
#include "PVertex.h"
#include "Mesh.h"
#include "D3DUtil.h"

BufferFactory::BufferFactory(ID3D11Device* p_device, ID3D11DeviceContext* p_deviceContext)
{
	m_device = p_device;
	m_deviceContext = p_deviceContext;
	m_elementSize = sizeof(float)*4;
}

BufferFactory::~BufferFactory()
{

}

Buffer<Mat4CBuffer>* BufferFactory::createMat4CBuffer()
{
	Buffer<Mat4CBuffer>* cBuffer;
	/// initialization data
	Mat4CBuffer data={
		{1.0f,0.0f,0.0f,  0.0f, // this here is an identity matrix
		0.0f,1.0f,0.0f,   0.0f,
		0.0f,0.0f,1.0f,   0.0f,
		0.0f,0.0f,0.0f,   1.0f}
	};

	// set up buffer description: usage, type and size
	BufferConfig::BUFFER_INIT_DESC bufferDesc;
	bufferDesc.ElementSize = m_elementSize;
	bufferDesc.Usage = BufferConfig::BUFFER_CPU_WRITE_DISCARD;
	bufferDesc.NumElements = sizeof(data)/m_elementSize;
	bufferDesc.Type = BufferConfig::CONSTANT_BUFFER_VS_PS;
	bufferDesc.Slot = BufferConfig::PERFRAME;


	// create and return the buffer
	cBuffer = new Buffer<Mat4CBuffer>(m_device,m_deviceContext,&data,bufferDesc);
	return cBuffer;
}


Buffer<InstanceData>* BufferFactory::createMat4InstanceBuffer( void* p_instanceList, 
															  unsigned int p_numberOfElements)
{
	Buffer<InstanceData>* instanceBuffer;

	// Create description for buffer
	BufferConfig::BUFFER_INIT_DESC bufferDesc;
	bufferDesc.ElementSize = sizeof(InstanceData);
	bufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
	bufferDesc.NumElements = p_numberOfElements;
	bufferDesc.Type = BufferConfig::VERTEX_BUFFER;
	bufferDesc.Slot = BufferConfig::SLOT0;

	// Create buffer from config and data
	instanceBuffer = new Buffer<InstanceData>(m_device,m_deviceContext,
		(InstanceData*)p_instanceList,bufferDesc);

	return instanceBuffer;
}

Buffer<PVertex>* BufferFactory::createFullScreenQuadBuffer()
{
	PVertex mesh[]= {
		{ 1,	-1,	0},
		{ -1,	-1,	0},
		{ 1,	1,	0},

		{ -1, -1,	0},
		{ 1,	1,	0},
		{ -1,	1,	0}
	};
	Buffer<PVertex>* quadBuffer;

	// Create description for buffer
	BufferConfig::BUFFER_INIT_DESC bufferDesc;
	bufferDesc.ElementSize = sizeof(PVertex);
	bufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
	bufferDesc.NumElements = 6;
	bufferDesc.Type = BufferConfig::VERTEX_BUFFER;
	bufferDesc.Slot = BufferConfig::SLOT0;

	// Create buffer from config and data
	quadBuffer = new Buffer<PVertex>(m_device,m_deviceContext,&mesh[0],bufferDesc);
	SETDEBUGNAME((quadBuffer->getBufferPointer()),("fullscreenQuad_buffer"));

	return quadBuffer;
}

Mesh* BufferFactory::createBoxMesh()
{

#pragma region static data
	PVertex mesh[]= {
		{-1,-1,-1},	
		{-1,1,-1},	
		{1,1,-1},	
		{1,-1,-1},	

		{-1,-1,1},	
		{1,-1,1},	
		{1,1,1},	
		{-1,1,1},	

		{-1,1,-1},	
		{-1,1,1},	
		{1,1,1},	
		{1,1,-1},	

		{-1,-1,-1},	
		{1,-1,-1},	
		{1,-1,1},	
		{-1,-1,1},	

		{-1,-1,1},	
		{-1,1,1},	
		{-1,1,-1},	
		{-1,-1,-1},	

		{1,-1,-1},	
		{1,1,-1},	
		{1,1,1},	
		{1,-1,1},	
	};

	unsigned int indices[] = {
		0,1,2,
		0,2,3,

		4,5,6,
		4,6,7,

		8,9,10,
		8,10,11,

		12,13,14,
		12,14,15,

		16,17,18,
		16,18,19,

		20,21,22,
		20,22,23
	};

#pragma endregion

	Mesh* newBox = new Mesh(createVertexBuffer(&mesh[0],
		sizeof(mesh)/sizeof(PVertex)),
		createIndexBuffer(&indices[0],
		sizeof(indices)/sizeof(unsigned int)));

	return newBox;
}


Buffer<unsigned int>* BufferFactory::createIndexBuffer( unsigned int* p_indices, 
												 unsigned int p_numberOfElements )
{	
	Buffer<unsigned int>* indexBuffer;

	// Create description for buffer
	BufferConfig::BUFFER_INIT_DESC indexBufferDesc;
	indexBufferDesc.ElementSize = sizeof(unsigned int);
	indexBufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
	indexBufferDesc.NumElements = p_numberOfElements;
	indexBufferDesc.Type = BufferConfig::INDEX_BUFFER;
	indexBufferDesc.Slot = BufferConfig::SLOT0;

	indexBuffer = new Buffer<unsigned int>(m_device,m_deviceContext, p_indices,
									 indexBufferDesc);

	return indexBuffer;
}
