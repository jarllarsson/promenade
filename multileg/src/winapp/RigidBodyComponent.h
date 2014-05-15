#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>

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
	btCollisionShape* m_collisionShape = new btSphereShape(1);
	btDefaultMotionState* m_motionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1),
		btVector3(0, -100, 0)));
	// Create rigidbody for ground
	// Bullet considers passing a mass of zero equivalent to making a body with infinite mass - it is immovable.
	btRigidBody::btRigidBodyConstructionInfo groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));
	btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
	// Add ground to world
	dynamicsWorld->addRigidBody(groundRigidBody);

	RigidBodyComponent(float posX, float posY)
	{
		this->posX = posX;
		this->posY = posY;
	};

	~RigidBodyComponent()
	{

	}
};