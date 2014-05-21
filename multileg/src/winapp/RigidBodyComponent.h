#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"
#include <glm\gtc\matrix_transform.hpp>
#include "ConstraintComponent.h"

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
		for (int i = 0; i < m_childConstraints.size();i++)
			if (m_childConstraints[i] != NULL) m_childConstraints[i]->forceRemove(m_dynamicsWorldPtr);
		delete m_collisionShape;
		delete m_rigidBody->getMotionState();
		delete m_rigidBody;
	}


	void init(unsigned int p_uid, btRigidBody* p_rigidBody, btDiscreteDynamicsWorld* p_dynamicsWorldPtr);

	float getMass();

	btCollisionShape* getCollisionShape();
	btRigidBody* getRigidBody();

	// These are set if another entity has this component's entity as parent
	// In case the parent is removed before the child, as we always need to remove
	// the constraint before any rigidbodies which it has references to. YIKES
	void addChildConstraint(ConstraintComponent* p_constraint)
	{
		m_childConstraints.push_back(p_constraint);
	}

	ConstraintComponent* getChildConstraint(unsigned int p_idx=0)
	{
		if (m_childConstraints.size() > 0 && m_childConstraints.size()>p_idx)
			return m_childConstraints[p_idx];
		else
			return NULL;
	}

	bool isInited();

	unsigned int getUID();

private:
	btCollisionShape* m_collisionShape;
	btRigidBody* m_rigidBody;
	vector<ConstraintComponent*> m_childConstraints;
	float m_mass;
	btDiscreteDynamicsWorld* m_dynamicsWorldPtr;
	unsigned int m_uid; ///< Unique id that can be used to retrieve this bodys entity from the rigidbodysystem
	bool m_inited; ///< initialized into the bullet physics world
};


// Init called by system on start
void RigidBodyComponent::init(unsigned int p_uid, btRigidBody* p_rigidBody, btDiscreteDynamicsWorld* p_dynamicsWorldPtr)
{
	m_rigidBody = p_rigidBody;
	m_dynamicsWorldPtr = p_dynamicsWorldPtr;
	m_inited = true;
	m_uid = p_uid;
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

unsigned int RigidBodyComponent::getUID()
{
	return m_uid;
}
