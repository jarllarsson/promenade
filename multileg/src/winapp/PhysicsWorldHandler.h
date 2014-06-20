#pragma once
#include <btBulletDynamicsCommon.h>
#include "ControllerSystem.h"

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
	PhysicsWorldHandler(btDynamicsWorld* p_world, ControllerSystem* p_controllerSystem)
	{
		m_world = p_world;
		m_controllerSystem = p_controllerSystem;
		m_world->setInternalTickCallback(physicsSimulationTickCallback, static_cast<void *>(this), true);
		m_internalStepCounter = 0;
	}

	void myProcessCallback(btScalar timeStep) 
	{
		m_internalStepCounter++;
		//// Character controller
		m_controllerSystem->fixedUpdate((float)timeStep); // might want this in post tick instead? Have it here for now
		//// Physics
		//btCollisionObjectArray objects = m_world->getCollisionObjectArray();
		//m_world->clearForces();
		//for (int i = 0; i < objects.size(); i++) {
		//	btRigidBody *rigidBody = btRigidBody::upcast(objects[i]);
		//	if (!rigidBody) {
		//		continue;
		//	}
		//	rigidBody->applyGravity();
		//	//rigidBody->applyTorque(btVector3(0.0f, 0.0f, 0.0f));
		//	//rigidBody->applyForce(btVector3(-10., 0., 0.), btVector3(0., 0., 0.));
		//}
		//// Controller
		m_controllerSystem->finish();
		m_controllerSystem->applyTorques((float)timeStep);
		// Other systems
		processSystemCollection((float)timeStep);
		return;
	}
	unsigned int getNumberOfInternalSteps()
	{
		return m_internalStepCounter;
	}

	void addOrderIndependentSystem(AdvancedEntitySystem* p_system)
	{
		m_orderIndependentSystems.push_back(p_system);
	}

	void processSystemCollection(float p_dt)
	{
		unsigned int count = (unsigned int)m_orderIndependentSystems.size();
		for (unsigned int i = 0; i < count; i++)
		{
			AdvancedEntitySystem* system = m_orderIndependentSystems[i];
			system->fixedUpdate(p_dt);
		}
	}
protected:
	// Might want to change this to generic list of a common base class
	ControllerSystem* m_controllerSystem; // But right now, we only need it for the controllers

	vector<AdvancedEntitySystem*> m_orderIndependentSystems;
	//
	// Physics world
	btDynamicsWorld* m_world;
	unsigned int m_internalStepCounter;
};

void physicsSimulationTickCallback(btDynamicsWorld *world, btScalar timeStep) {
	PhysicsWorldHandler *w = static_cast<PhysicsWorldHandler *>(world->getWorldUserInfo());
	w->myProcessCallback(timeStep);
}