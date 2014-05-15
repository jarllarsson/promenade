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
		delete m_collisionShape;
		delete m_rigidBody->getMotionState();
		delete m_rigidBody;
	}

	void init(TransformComponent* p_transform);

	float getMass();


	bool isInited();

private:
	float m_mass;

	bool m_inited; ///< initialized into the bullet physics world
};


// Init called by system on start
void RigidBodyComponent::init(TransformComponent* p_transform)
{
	m_inited = true;
	// Set up the rigidbody
	glm::vec3 position = p_transform->getPosition();
	glm::quat rotation = p_transform->getRotation();
	btDefaultMotionState* motionState = new btDefaultMotionState(btTransform(btQuaternion(rotation.x, rotation.y, rotation.z, rotation.w),
																 btVector3(position.x, position.y, position.z)));
	// Calculate inertia, using our collision shape
	btVector3 inertia(0, 0, 0);
	m_collisionShape->calculateLocalInertia(m_mass, inertia);
	// Construction info
	btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(m_mass, motionState, m_collisionShape, inertia);
	m_rigidBody = new btRigidBody(rigidBodyCI);
}

bool RigidBodyComponent::isInited()
{
	return m_inited;
}

float RigidBodyComponent::getMass()
{
	return m_mass;
}
