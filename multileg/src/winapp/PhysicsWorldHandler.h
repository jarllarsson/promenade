#pragma once
#include <btBulletDynamicsCommon.h>

// =======================================================================================
//                                      PhysicsWorldHandler
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Handles the simulation tick callback for applying forces and torques etc
///         Why this is needed: 
///			http://hub.jmonkeyengine.org/forum/topic/using-applyforce-with-bullet-on-android/
/// # PhysicsWorldHandler
/// 
/// 19-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

static void physicsSimulationTickCallback(btDynamicsWorld *world, btScalar timeStep);

class PhysicsWorldHandler {
public:
	PhysicsWorldHandler(btDynamicsWorld* p_world)
	{
		m_world = p_world;
		m_world->setInternalTickCallback(physicsSimulationTickCallback, static_cast<void *>(this), true);
	}

	void myProcessCallback(btScalar timeStep) {
		btCollisionObjectArray objects = m_world->getCollisionObjectArray();
		m_world->clearForces();
		for (int i = 0; i < objects.size(); i++) {
			btRigidBody *rigidBody = btRigidBody::upcast(objects[i]);
			if (!rigidBody) {
				continue;
			}
			rigidBody->applyGravity();
			rigidBody->applyForce(btVector3(-10., 0., 0.), btVector3(0., 0., 0.));
		}
		return;
	}

protected:
	btDynamicsWorld* m_world;
};

void physicsSimulationTickCallback(btDynamicsWorld *world, btScalar timeStep) {
	PhysicsWorldHandler *w = static_cast<PhysicsWorldHandler *>(world->getWorldUserInfo());
	w->myProcessCallback(timeStep);
}