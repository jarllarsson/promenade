#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"

// =======================================================================================
//                                RigidBodyComponent
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # RigidBodyComponent
/// 
/// 15-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class RigidBodyComponent : public artemis::Component
{
public:
	btCollisionShape* m_collisionShape;
	btDefaultMotionState* m_motionState;
	btRigidBody* m_rigidBody;

	RigidBodyComponent(btCollisionShape* p_collisionShape = NULL, float p_mass=1.0f)
	{
		if (p_collisionShape == NULL)
		{
			m_collisionShape = new btSphereShape(1);
		}
		else
		{
			m_collisionShape = p_collisionShape;
		}
		m_mass = p_mass;
	};

	~RigidBodyComponent()
	{
		m_dynamicsWorldPtr->removeRigidBody(m_rigidBody);
		delete m_collisionShape;
		delete m_motionState;
		delete m_rigidBody;
	}

	void init(btDiscreteDynamicsWorld* p_world, TransformComponent* p_transform);

	float getMass();


	bool isInited();

private:
	float m_mass;

	bool m_inited; ///< initialized into the bullet physics world
	btDiscreteDynamicsWorld* m_dynamicsWorldPtr;
};


// Init called by system on start
void RigidBodyComponent::init(btDiscreteDynamicsWorld* p_world, TransformComponent* p_transform)
{
	m_inited = true;
	m_dynamicsWorldPtr = p_world;
	// Set up the rigidbody
	glm::vec3 position = p_transform->getPosition();
	m_motionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(position.x, position.y, position.z)));
	// Calculate inertia, using our collision shape
	btVector3 inertia(0, 0, 0);
	m_collisionShape->calculateLocalInertia(m_mass, inertia);
	// Construction info
	btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(m_mass, m_motionState, m_collisionShape, inertia);
	m_rigidBody = new btRigidBody(rigidBodyCI);
	//
	// Add to world
	m_dynamicsWorldPtr->addRigidBody(m_rigidBody);
}

bool RigidBodyComponent::isInited()
{
	return m_inited;
}

float RigidBodyComponent::getMass()
{
	return m_mass;
}
