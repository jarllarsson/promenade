#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"
#include <glm\gtc\matrix_transform.hpp>

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
		m_rigidBody = NULL;
		m_mass = p_mass;
		m_inited = false;
	};

	virtual ~RigidBodyComponent()
	{
		delete m_collisionShape;
		delete m_rigidBody->getMotionState();
		delete m_rigidBody;
	}

	void init(btRigidBody* p_rigidBody);

	float getMass();

	btCollisionShape* getCollisionShape();
	btRigidBody* getRigidBody();

	bool isInited();

private:
	btCollisionShape* m_collisionShape;
	btRigidBody* m_rigidBody;
	float m_mass;

	bool m_inited; ///< initialized into the bullet physics world
};


// Init called by system on start
void RigidBodyComponent::init( btRigidBody* p_rigidBody )
{
	m_rigidBody = p_rigidBody;
	m_inited = true;
}

bool RigidBodyComponent::isInited()
{
	return m_inited;
}

float RigidBodyComponent::getMass()
{
	return m_mass;
}

btCollisionShape* RigidBodyComponent::getCollisionShape()
{
	return m_collisionShape;
}

btRigidBody* RigidBodyComponent::getRigidBody()
{
	return m_rigidBody;
}
