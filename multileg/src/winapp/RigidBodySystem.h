#pragma once

#include <Artemis.h>
#include "TransformComponent.h"
#include "RigidBodyComponent.h"
#include <btBulletDynamicsCommon.h>

// =======================================================================================
//                                      RigidBodySystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Updates transforms based on the result on rigidbodies
///        
/// # RigidBodySystem
/// 
/// 15-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class RigidBodySystem : public artemis::EntityProcessingSystem
{
private:
	artemis::ComponentMapper<TransformComponent> transformMapper;
	artemis::ComponentMapper<RigidBodyComponent> rigidBodyMapper;
	btDiscreteDynamicsWorld* m_dynamicsWorldPtr;
public:
	RigidBodySystem(btDiscreteDynamicsWorld* p_dynamicsWorld) 
	{
		addComponentType<TransformComponent>();
		addComponentType<RigidBodyComponent>();
		m_dynamicsWorldPtr = p_dynamicsWorld;
	};

	virtual void initialize() 
	{
		transformMapper.init(*world);
		rigidBodyMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e)
	{
		RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
		if (rigidBody->isInited())
		{
			m_dynamicsWorldPtr->removeRigidBody(rigidBody->m_rigidBody);
		}
	};

	virtual void added(artemis::Entity &e)
	{
		RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
		TransformComponent* transform = transformMapper.get(e);
		if (!rigidBody->isInited())
		{
			rigidBody->init(transform);
			m_dynamicsWorldPtr->addRigidBody(rigidBody->m_rigidBody);
		}
	};


	virtual void processEntity(artemis::Entity &e) 
	{
		RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
		TransformComponent* transform = transformMapper.get(e);
		// Update transform of object
		if (rigidBody->isInited())
		{
			btRigidBody* body = rigidBody->m_rigidBody;
			btMotionState* motionState = body->getMotionState();
			btTransform transform;
			motionState->getWorldTransform(transform);
		}
	};

};