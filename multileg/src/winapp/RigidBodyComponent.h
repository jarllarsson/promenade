#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"
#include <glm\gtc\matrix_transform.hpp>
#include "ConstraintComponent.h"
#include <vector>

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

	RigidBodyComponent(btCollisionShape* p_collisionShape = NULL, float p_mass = 1.0f);

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

	bool isInited();

private:
	btCollisionShape* m_collisionShape;
	btRigidBody* m_rigidBody;
	std::vector<ConstraintComponent*> m_childConstraints;
	float m_mass;
	btDiscreteDynamicsWorld* m_dynamicsWorldPtr;
	unsigned int m_uid; ///< Unique id that can be used to retrieve this bodys entity from the rigidbodysystem
	bool m_inited; ///< initialized into the bullet physics world
};

