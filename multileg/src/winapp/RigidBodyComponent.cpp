#include "RigidBodyComponent.h"
#include <ToString.h>
#include <DebugPrint.h>


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
	m_linearFactor = glm::vec3(1, 1, 1);
	m_angularFactor = glm::vec3(1, 1, 1);
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
	m_linearFactor = glm::vec3(1, 1, 1);
	m_angularFactor = glm::vec3(1, 1, 1);
}

RigidBodyComponent::~RigidBodyComponent()
{
	if (m_inited)
	{
		/*unsigned int csz = m_childConstraints.size();
		DEBUGPRINT(((" sz:"+ToString(csz)).c_str()));
		if (csz > 0)
		{
			for (unsigned int i = 0; i < csz; i++)
			{
				DEBUGPRINT(((" p:"+ToString(m_childConstraints[i])).c_str()));
				if (m_childConstraints[i] != NULL)
				{
					std::string strr = m_childConstraints[i]->getDesc()->m_collisionBetweenLinked ? "[T]" : "[F]";
					DEBUGPRINT(((" cc:" + strr).c_str()));
					m_childConstraints[i]->forceRemove(m_dynamicsWorldPtr);
				}
			}
		}*/

		m_childConstraints.clear();
		SAFE_DELETE(m_collisionShape);
		if (m_rigidBody != NULL && m_rigidBody->getMotionState()) delete m_rigidBody->getMotionState();
		SAFE_DELETE(m_callback);
		if (m_dynamicsWorldPtr!=NULL) m_dynamicsWorldPtr->removeRigidBody(m_rigidBody);
	}
	m_inited = false;
	SAFE_DELETE(m_rigidBody);
}


void RigidBodyComponent::init(unsigned int p_uid, btRigidBody* p_rigidBody, 
	btDiscreteDynamicsWorld* p_dynamicsWorldPtr)
{
	m_rigidBody = p_rigidBody;
	m_dynamicsWorldPtr = p_dynamicsWorldPtr;
	m_inited = true;
	m_uid = p_uid;
	setLinearFactor(m_linearFactor);
	setAngularFactor(m_angularFactor);
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

const glm::vec3& RigidBodyComponent::getVelocity()
{
	return m_velocity;
}

const glm::vec3& RigidBodyComponent::getAcceleration()
{
	return m_acceleration;
}

void RigidBodyComponent::setVelocityStat(glm::vec3& p_velocity)
{
	m_velocity = p_velocity;
}

void RigidBodyComponent::setAccelerationStat(glm::vec3& p_acceleration)
{
	m_acceleration = p_acceleration;
}

const glm::vec3& RigidBodyComponent::getLinearFactor()
{
	return m_linearFactor;
}

const glm::vec3& RigidBodyComponent::getAngularFactor()
{
	return m_angularFactor;
}

void RigidBodyComponent::setLinearFactor(glm::vec3& p_axis)
{
	m_linearFactor = p_axis;
	if (m_rigidBody != NULL)
		m_rigidBody->setLinearFactor(btVector3(m_linearFactor.x, m_linearFactor.y, m_linearFactor.z));
}

void RigidBodyComponent::setAngularFactor(glm::vec3& p_axis)
{
	m_angularFactor = p_axis;
	if (m_rigidBody != NULL)
		m_rigidBody->setAngularFactor(btVector3(m_angularFactor.x, m_angularFactor.y, m_angularFactor.z));
}
