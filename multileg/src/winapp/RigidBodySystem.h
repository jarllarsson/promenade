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
			//DEBUGPRINT(((string("Init rigidbody m=") + toString(rigidBody->getMass()) + "\n").c_str()));
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
			if (body!=NULL/* && body->isInWorld() && body->isActive()*/)
			{			
				btMotionState* motionState = body->getMotionState();
				btTransform physTransform;
				motionState->getWorldTransform(physTransform);
				// update the transform component
				// Get the transform from Bullet and into mat
				glm::mat4 mat;
				// first we need to keep scale as bullet doesn't
				glm::vec3 scale = transform->getScale();
				physTransform.getOpenGLMatrix(glm::value_ptr(mat));
				transform->setMatrix(mat);
				transform->setScaleToMatrix(scale);
				//btVector3 pos = physTransform.getOrigin();
				//btQuaternion rot = physTransform.getRotation();
				////DEBUGPRINT(((string("run rigidbody py=") + toString(pos.y()) + "\n").c_str()));
				//transform->setPosRotToMatrix(glm::vec3(pos.x(), pos.y(), pos.z()),
				//						     glm::quat(rot.x(), rot.y(), rot.z(), rot.w()));
			}
		}
	};

};