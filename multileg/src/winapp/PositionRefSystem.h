#pragma once

#include <Artemis.h>
#include <GraphicsDevice.h>
#include <BufferFactory.h>
#include <Util.h>
#include "TransformComponent.h"
#include "RenderComponent.h"
#include "MaterialComponent.h"
#include "PositionRefComponent.h"



// =======================================================================================
//                                    PositionRefSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # RenderSystem
/// 
/// 23-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class PositionRefSystem : public artemis::EntityProcessingSystem
{
private:
	artemis::ComponentMapper<TransformComponent> transformMapper;
	artemis::ComponentMapper<PositionRefComponent> posRefMapper;
public:
	PositionRefSystem()
	{
		addComponentType<TransformComponent>();
		addComponentType<PositionRefComponent>();
	};

	virtual ~PositionRefSystem()
	{

	}

	virtual void initialize()
	{
		transformMapper.init(*world);
		posRefMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e)
	{

	};

	virtual void added(artemis::Entity &e)
	{

	};

	virtual void processEntity(artemis::Entity &e)
	{
		PositionRefComponent* posRef = posRefMapper.get(e);
		TransformComponent* transform = transformMapper.get(e);
		if (posRef != NULL && transform != NULL)
		{
			transform->setPositionToMatrix(*posRef->m_posRef);
		}
	};

	virtual void end()
	{

	};
};