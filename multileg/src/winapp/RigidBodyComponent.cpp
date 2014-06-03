#include "RigidBodyComponent.h"


RigidBodyComponent::RigidBodyComponent(btCollisionShape* p_collisionShape /*= NULL*/,
	float p_mass /*= 1.0f*/)
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
}

RigidBodyComponent::~RigidBodyComponent()
{
	for (int i = 0; i < m_childConstraints.size(); i++)
		if (m_childConstraints[i] != NULL) m_childConstraints[i]->forceRemove(m_dynamicsWorldPtr);
	delete m_collisionShape;
	delete m_rigidBody->getMotionState();
	delete m_rigidBody;
}


void RigidBodyComponent::init(unsigned int p_uid, btRigidBody* p_rigidBody, 
	btDiscreteDynamicsWorld* p_dynamicsWorldPtr)
{
	m_rigidBody = p_rigidBody;
	m_dynamicsWorldPtr = p_dynamicsWorldPtr;
	m_inited = true;
	m_uid = p_uid;
}

btRigidBody* RigidBodyComponent::getRigidBody()
{
	return m_rigidBody;
}

void RigidBodyComponent::addChildConstraint(ConstraintComponent* p_constraint)
{
	m_childConstraints.push_back(p_constraint);
}


ConstraintComponent* RigidBodyComponent::getChildConstraint(unsigned int p_idx/* = 0*/)
{
	if (m_childConstraints.size() > 0 && m_childConstraints.size() > p_idx)
		return m_childConstraints[p_idx];
	else
		return NULL;
}


float RigidBodyComponent::getMass()
{
	return m_mass;
}

btCollisionShape* RigidBodyComponent::getCollisionShape()
{
	return m_collisionShape;
}

bool RigidBodyComponent::isInited()
{
	return m_inited;
}

unsigned int RigidBodyComponent::getUID()
{
	return m_uid;
}
