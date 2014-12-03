#include "PhysicsWorldHandler.h"
#include <btBulletDynamicsCommon.h>
#include "AdvancedEntitySystem.h"
#include "ControllerSystem.h"
#include <DebugPrint.h>
#include <ToString.h>


void physicsSimulationTickCallback(btDynamicsWorld *world, btScalar timeStep) {
	PhysicsWorldHandler *w = static_cast<PhysicsWorldHandler *>(world->getWorldUserInfo());
	w->physProcessCallback(timeStep);
}

PhysicsWorldHandler::PhysicsWorldHandler(btDynamicsWorld* p_world, ControllerSystem* p_controllerSystem)
{
	m_world = p_world;
	m_controllerSystem = p_controllerSystem;
	m_world->setInternalTickCallback(physicsSimulationTickCallback, static_cast<void *>(this), true);
	m_internalStepCounter = 0;
}

void PhysicsWorldHandler::physProcessCallback(btScalar timeStep)
{
	int numManifolds = m_world->getDispatcher()->getNumManifolds();
	DEBUGPRINT(( ("\ncollision! n:"+ToString(numManifolds)).c_str() ));

	for (int i = 0; i < numManifolds; i++)
	{
		btPersistentManifold* contactManifold = m_world->getDispatcher()->getManifoldByIndexInternal(i);
		const btCollisionObject* obA = static_cast<const btCollisionObject*>(contactManifold->getBody0());
		const btCollisionObject* obB = static_cast<const btCollisionObject*>(contactManifold->getBody1());

		int numContacts = contactManifold->getNumContacts();
		for (int j = 0; j < numContacts; j++)
		{
			btManifoldPoint& pt = contactManifold->getContactPoint(j);
			if (pt.getDistance() < 0.f)
			{
				// get user pointer/index
				const btVector3& ptA = pt.getPositionWorldOnA();
				const btVector3& ptB = pt.getPositionWorldOnB();
				const btVector3& normalOnB = pt.m_normalWorldOnB;
				DEBUGPRINT((("\nbd" + ToString(obA) + "+bd" + ToString(obB)).c_str()));
			}
		}
	}

	m_internalStepCounter++;
	processPreprocessSystemCollection((float)timeStep);
	//// Character controller
	m_controllerSystem->fixedUpdate((float)timeStep); // might want this in post tick instead? Have it here for now
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
