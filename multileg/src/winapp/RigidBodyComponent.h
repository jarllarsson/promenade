#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"
#include <glm\gtc\matrix_transform.hpp>
#include "ConstraintComponent.h"
#include <vector>
#include "CollisionLayer.h"

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
	enum ListenerMode
	{
		REGISTER_COLLISIONS,
		DONT_REGISTER_COLLISIONS,
	};

	RigidBodyComponent(btCollisionShape* p_collisionShape = NULL, float p_mass = 1.0f, 
		short int p_collisionLayerType=CollisionLayer::CollisionLayerType::COL_DEFAULT,
		short int p_collidesWithLayer = CollisionLayer::CollisionLayerType::COL_DEFAULT);	
	
	RigidBodyComponent(ListenerMode p_registerCollisions, btCollisionShape* p_collisionShape = NULL, float p_mass = 1.0f,
		short int p_collisionLayerType = CollisionLayer::CollisionLayerType::COL_DEFAULT,
		short int p_collidesWithLayer = CollisionLayer::CollisionLayerType::COL_DEFAULT);

	virtual ~RigidBodyComponent();


	// Init called by system on start
	void init(unsigned int p_uid, btRigidBody* p_rigidBody, btDiscreteDynamicsWorld* p_dynamicsWorldPtr);
	// getters
	float					getMass();
	btCollisionShape*		getCollisionShape();
	btRigidBody*			getRigidBody();
	ConstraintComponent*	getChildConstraint(unsigned int p_idx = 0);
	unsigned int			getUID();

	// These are set if another entity has this component's entity as parent
	// In case the parent is removed before the child, as we always need to remove
	// the constraint before any rigidbodies which it has references to. YIKES
	void addChildConstraint(ConstraintComponent* p_constraint);

	void addCollisionCallback(btCollisionWorld::ContactResultCallback* p_callback);
	btCollisionWorld::ContactResultCallback* getCollisionCallbackFunc();

	bool isColliding();
	const glm::vec3& getCollisionPoint();
	const glm::vec3& getVelocity();
	const glm::vec3& getAcceleration();
	const glm::vec3& getLinearFactor();
	const glm::vec3& getAngularFactor();
	bool isRegisteringCollisions();
	void unsetIsCollidingFlag();
	void setCollidingStat(bool p_stat, const glm::vec3& p_position=glm::vec3(0.0f));
	void setVelocityStat(glm::vec3& p_velocity);
	void setAccelerationStat(glm::vec3& p_acceleration);
	void setLinearFactor(glm::vec3& p_axis);
	void setAngularFactor(glm::vec3& p_axis);

	bool isInited();
	short int m_collisionLayerType;
	short int m_collidesWithLayer;
private:
	glm::vec3 m_linearFactor; // allowed axes for movement
	glm::vec3 m_angularFactor;// allowed axes for rotation
	glm::vec3 m_collisionPoint; 
	glm::vec3 m_velocity;
	glm::vec3 m_acceleration;
	bool m_registerCollisions;
	bool m_colliding;
	btCollisionShape* m_collisionShape;
	btRigidBody* m_rigidBody;
	std::vector<ConstraintComponent*> m_childConstraints;
	btCollisionWorld::ContactResultCallback* m_callback;
	float m_mass;
	btDiscreteDynamicsWorld* m_dynamicsWorldPtr;
	unsigned int m_uid; ///< Unique id that can be used to retrieve this bodys entity from the rigidbodysystem
	bool m_inited; ///< initialized into the bullet physics world
};

