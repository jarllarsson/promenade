#include "RigidBodyComponent.h"


RigidBodyComponent::RigidBodyComponent(btCollisionShape* p_collisionShape /*= NULL*/,
	float p_mass /*= 1.0f*/, 
	short int p_collisionLayerType/*=CollisionLayer::CollisionLayerType::COL_DEFAULT*/,
	short int p_collidesWithLayer/*=CollisionLayer::CollisionLayerType::COL_DEFAULT*/)
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
	m_collisionLayerType = p_collisionLayerType;
	m_collidesWithLayer = p_collidesWithLayer;
	m_callback = NULL;
	m_registerCollisions = false;
	m_colliding = false;
}

RigidBodyComponent::RigidBodyComponent(ListenerMode p_registerCollisions, 
	btCollisionShape* p_collisionShape /*= NULL*/,
	float p_mass /*= 1.0f*/, 
	short int p_collisionLayerType /*= CollisionLayer::CollisionLayerType::COL_DEFAULT*/, 
	short int p_collidesWithLayer /*= CollisionLayer::CollisionLayerType::COL_DEFAULT*/)
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
	m_collisionLayerType = p_collisionLayerType;
	m_collidesWithLayer = p_collidesWithLayer;
	m_callback = NULL;
	m_registerCollisions = (p_registerCollisions==ListenerMode::REGISTER_COLLISIONS);
	m_colliding = false;
}

RigidBodyComponent::~RigidBodyComponent()
{
	for (unsigned int i = 0; i < m_childConstraints.size(); i++)
		if (m_childConstraints[i] != NULL) m_childConstraints[i]->forceRemove(m_dynamicsWorldPtr);
	delete m_collisionShape;
	if (m_rigidBody!=NULL) delete m_rigidBody->getMotionState();
	if (m_callback != NULL) delete m_callback;
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

void RigidBodyComponent::addCollisionCallback(btCollisionWorld::ContactResultCallback* p_callback)
{
	m_callback = p_callback;
}

btCollisionWorld::ContactResultCallback* RigidBodyComponent::getCollisionCallbackFunc()
{
	return m_callback;
}

bool RigidBodyComponent::isRegisteringCollisions()
{
	return m_registerCollisions;
}


void RigidBodyComponent::setCollidingStat(bool p_stat, const glm::vec3& p_position/* = glm::vec3(0.0f)*/)
{
	m_colliding = p_stat;
	m_collisionPoint = p_position;
}

const glm::vec3& RigidBodyComponent::getCollisionPoint()
{
	return m_collisionPoint;
}

bool RigidBodyComponent::isColliding()
{
	return m_colliding;
}
