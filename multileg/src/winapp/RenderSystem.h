#pragma once

#include <Artemis.h>
#include <GraphicsDevice.h>
#include <BufferFactory.h>
#include <Util.h>
#include "TransformComponent.h"
#include "RenderComponent.h"



// =======================================================================================
//                                      RenderSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # RenderSystem
/// 
/// 16-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class RenderSystem : public artemis::EntityProcessingSystem
{
private:
	artemis::ComponentMapper<TransformComponent> transformMapper;
	artemis::ComponentMapper<RenderComponent> renderMapper;
	GraphicsDevice* m_graphicsDevice;

	// Real resource handler ref here later:
	Buffer<glm::mat4>* m_instances;

	// render stats
	bool m_instancesUpdated;
public:
	RenderSystem(GraphicsDevice* p_graphicsDevice, Buffer<glm::mat4>* p_instances)
	{
		addComponentType<TransformComponent>();
		addComponentType<RenderComponent>();
		m_graphicsDevice = p_graphicsDevice;
		m_instances = p_instances;
		m_instancesUpdated = true;
	};

	virtual void initialize()
	{
		transformMapper.init(*world);
		renderMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e)
	{
		RenderComponent* renderStats = renderMapper.get(e);
		// some smart removal here
		// also, make adding smarter, it's super retarded now
	};

	virtual void added(artemis::Entity &e)
	{
		RenderComponent* renderStats = renderMapper.get(e);
		TransformComponent* transform = transformMapper.get(e);
		// warning quick and dirty
		// If new entity added with rendercomponent
		// resize instance array, ie. destroy it create new with new size, add new value
		glm::mat4* instances = m_instances->accessBufferArr;
		unsigned int arraySize = m_instances->getArraySize();
		unsigned int newArraySize = arraySize+1;
		unsigned int backIdxNew = arraySize; // when we resize
		glm::mat4* newInstances = new glm::mat4[newArraySize]; // resize
		// add the matrix
		newInstances[backIdxNew] = glm::transpose(transform->getMatrix());
		// remove the old buffer
		SAFE_DELETE(m_instances);
		// recreate the buffer
		m_instances = m_graphicsDevice->getBufferFactoryRef()->createMat4InstanceBuffer((void*)newInstances, newArraySize);
		DEBUGPRINT(( (string("add renderobj (ilist sz[") + toString(arraySize) + "] -> [" + toString(newArraySize)+"])\n").c_str() ));
		//
		m_instancesUpdated = true;
	};

	// update instance list based on transform
	virtual void processEntity(artemis::Entity &e)
	{
		RenderComponent* renderStats = renderMapper.get(e);
		TransformComponent* transform = transformMapper.get(e);
		if (transform->isTransformRenderDirty())
		{
			// Update transform of object to instance list buffer
			// gpu write to buffer is deferred for when all changes have been made
			// the deferred update is done in end()
			unsigned int instanceIdx = renderStats->getInstanceIdx();
			glm::mat4* writeMat = m_instances->readElementPtrAt(instanceIdx);
			const glm::mat4* readMat = transform->getMatrixPtr();
			if (writeMat != NULL)
			{
				*writeMat = glm::transpose(*readMat);
			}
			DEBUGPRINT(((string("render instance ") + toString(instanceIdx) + "\n").c_str()));
			transform->unsetTransformRenderDirty();
			m_instancesUpdated = true;
		}
	};

	virtual void end()
	{
		if (m_instancesUpdated)
		{
			m_instances->update(); // write all to buffer as one call
			m_instancesUpdated = false;
		}
	};

};