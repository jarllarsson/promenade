#pragma once

#include <Artemis.h>
#include "TransformComponent.h"
#include "RigidBodyComponent.h"
#include <btBulletDynamicsCommon.h>
#include "ConstraintComponent.h"
#include <vector>

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
	// Vector to store creation calls for constraints
	// This is used so they can be inited in the correct order
	std::vector<artemis::Entity*> m_constraintCreationsList;
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

	virtual void removed(artemis::Entity &e);

	virtual void added(artemis::Entity &e);

	virtual void processEntity(artemis::Entity &e);

	// Void this has to be called explicitly for it to be done correctly
	// constraints need both its rigidbodies to have been added to the physics world
	// ie. after all entity adds. I can't control the order of adds, unlike processing.
	void executeDeferredConstraintInits();

private:
	void checkForNewConstraints(artemis::Entity &e);
	void setupConstraints(artemis::Entity *e);

};

void RigidBodySystem::removed(artemis::Entity &e)
{
	RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
	if (rigidBody->isInited())
	{
		m_dynamicsWorldPtr->removeRigidBody(rigidBody->getRigidBody());
	}
};

void RigidBodySystem::added(artemis::Entity &e)
{
	RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
	TransformComponent* transform = transformMapper.get(e);
	if (!rigidBody->isInited())
	{
		// Set up the rigidbody
		btTransform t;
		// Construct matrix without scale (or the shape will bug)
		glm::mat4 translate = glm::translate(glm::mat4(1.0f), transform->getPosition());
		glm::mat4 rotate = glm::mat4_cast(transform->getRotation());
		glm::mat4 mat = translate * rotate;
		// Read to bt matrix
		t.setFromOpenGLMatrix(glm::value_ptr(mat));
		// Init motionstate with matrix
		btDefaultMotionState* motionState = new btDefaultMotionState(t);
		// Calculate inertia, using our collision shape
		btVector3 inertia(0, 0, 0);
		float mass = rigidBody->getMass();
		btCollisionShape* collisionShape = rigidBody->getCollisionShape();
		collisionShape->calculateLocalInertia(mass, inertia);
		// Construction info
		btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(mass, motionState, collisionShape, inertia);
		btRigidBody* rigidBodyInstance = new btRigidBody(rigidBodyCI);
		//
		rigidBody->init(rigidBodyInstance);
		m_dynamicsWorldPtr->addRigidBody(rigidBody->getRigidBody());
		// check if entity has constraints, if so, and if they're uninited, add to
		// list for batch init (as they must have both this entity's and parent's rb in phys world.
		checkForNewConstraints(e);
	}
};


void RigidBodySystem::processEntity(artemis::Entity &e)
{
	RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
	TransformComponent* transform = transformMapper.get(e);
	// Update transform of object
	if (rigidBody->isInited())
	{
		btRigidBody* body = rigidBody->getRigidBody();
		if (body != NULL/* && body->isInWorld() && body->isActive()*/)
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
		}
	}
}

void RigidBodySystem::checkForNewConstraints(artemis::Entity &e)
{
	ConstraintComponent* constraint = (ConstraintComponent*)e.getComponent<ConstraintComponent>();
	if (constraint != NULL && !constraint->isInited())
	{
		m_constraintCreationsList.push_back(&e);
	}
}

void RigidBodySystem::executeDeferredConstraintInits()
{
	for (int i = 0; i < m_constraintCreationsList.size(); i++)
	{
		setupConstraints(m_constraintCreationsList[i]);
	}
	m_constraintCreationsList.clear();
}

void RigidBodySystem::setupConstraints(artemis::Entity *e)
{
	// This (child) is guaranteed to have the right components
	// As it was already processed
	RigidBodyComponent* rigidBody = rigidBodyMapper.get(*e);
	TransformComponent* transform = transformMapper.get(*e);
	ConstraintComponent* constraint = (ConstraintComponent*)e->getComponent<ConstraintComponent>();
	if (constraint != NULL && !constraint->isInited() && constraint->getParent()!=NULL)
	{
		RigidBodyComponent* parentRigidBody = (RigidBodyComponent*)constraint->getParent()->getComponent<RigidBodyComponent>();
		TransformComponent* parentTransform = (TransformComponent*)constraint->getParent()->getComponent<TransformComponent>();
		if (parentRigidBody != NULL && parentTransform != NULL)
		{

		}
	}
}

