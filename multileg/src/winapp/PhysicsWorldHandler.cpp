#include "PhysicsWorldHandler.h"
#include <btBulletDynamicsCommon.h>
#include "AdvancedEntitySystem.h"
#include "ControllerSystem.h"


void physicsSimulationTickCallback(btDynamicsWorld *world, btScalar timeStep) {
	PhysicsWorldHandler *w = static_cast<PhysicsWorldHandler *>(world->getWorldUserInfo());
	w->myProcessCallback(timeStep);
}

PhysicsWorldHandler::PhysicsWorldHandler(btDynamicsWorld* p_world, ControllerSystem* p_controllerSystem)
{
	m_world = p_world;
	m_controllerSystem = p_controllerSystem;
	m_world->setInternalTickCallback(physicsSimulationTickCallback, static_cast<void *>(this), true);
	m_internalStepCounter = 0;
}

void PhysicsWorldHandler::myProcessCallback(btScalar timeStep)
{
	m_internalStepCounter++;
	processPreprocessSystemCollection((float)timeStep);
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
	processOrderIndependentSystemCollection((float)timeStep);
	return;
}
unsigned int PhysicsWorldHandler::getNumberOfInternalSteps()
{
	return m_internalStepCounter;
}

void PhysicsWorldHandler::addOrderIndependentSystem(AdvancedEntitySystem* p_system)
{
	m_orderIndependentSystems.push_back(p_system);
}

void PhysicsWorldHandler::processOrderIndependentSystemCollection(float p_dt)
{
	unsigned int count = (unsigned int)m_orderIndependentSystems.size();
	for (unsigned int i = 0; i < count; i++)
	{
		AdvancedEntitySystem* system = m_orderIndependentSystems[i];
		system->fixedUpdate(p_dt);
	}
}

void PhysicsWorldHandler::addPreprocessSystem(AdvancedEntitySystem* p_system)
{
	m_preprocessSystems.push_back(p_system);
}

void PhysicsWorldHandler::processPreprocessSystemCollection(float p_dt)
{
	unsigned int count = (unsigned int)m_preprocessSystems.size();
	for (unsigned int i = 0; i < count; i++)
	{
		AdvancedEntitySystem* system = m_preprocessSystems[i];
		system->fixedUpdate(p_dt);
	}
}
