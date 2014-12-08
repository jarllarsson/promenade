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
	m_internalStepCounter++;
	processPreprocessSystemCollection((float)timeStep);
	// Collisions readback for rigidbodies that has it enabled
	handleCollisions();
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

void PhysicsWorldHandler::handleCollisions()
{
	int numManifolds = m_world->getDispatcher()->getNumManifolds();
	//DEBUGPRINT((("\ncollision! n:" + ToString(numManifolds)).c_str()));

	for (int i = 0; i < numManifolds; i++)
	{
		btPersistentManifold* contactManifold = m_world->getDispatcher()->getManifoldByIndexInternal(i);
		const btCollisionObject* obA = contactManifold->getBody0();
		const btCollisionObject* obB = contactManifold->getBody1();

		RigidBodyComponent* rigidBodyA = (RigidBodyComponent*)obA->getUserPointer();
		RigidBodyComponent* rigidBodyB = (RigidBodyComponent*)obB->getUserPointer();
		
		// Get user pointers to rigid bodies if they exist
		// and see if collision check is activated
		bool rbACollision = false;
		if (rigidBodyA!=NULL) rbACollision=rigidBodyA->isRegisteringCollisions();
		bool rbBCollision = false;
		if (rigidBodyB != NULL) rbBCollision=rigidBodyB->isRegisteringCollisions();

		// If one has writeback enabled..
		if (rbACollision || rbBCollision)
		{
			int numContacts = contactManifold->getNumContacts();
			for (int j = 0; j < numContacts; j++)
			{
				btManifoldPoint& pt = contactManifold->getContactPoint(j);
				if (pt.getDistance() < 0.0f/* && checkMaskedCollision(obA, obB)*/)
				{
					const btVector3& ptA = pt.getPositionWorldOnA();
					const btVector3& ptB = pt.getPositionWorldOnB();
					const btVector3& normalOnB = pt.m_normalWorldOnB;
					if (rbACollision && rigidBodyA != NULL) rigidBodyA->setCollidingStat(true, glm::vec3(ptA.x(), ptA.y(), ptA.z()));
					if (rbBCollision && rigidBodyB != NULL) rigidBodyB->setCollidingStat(true, glm::vec3(ptB.x(), ptB.y(), ptB.z()));
					//DEBUGPRINT((("\nbd" + ToString(rigidBodyA->getUID()) + "+bd" + ToString(rigidBodyB->getUID())).c_str()));
				}
			}
		}
	}
}

bool PhysicsWorldHandler::checkMaskedCollision(const btCollisionObject* p_colObj0, const btCollisionObject* p_colObj1)
{
	CollisionLayer::CollisionLayerType obj0Grp = (CollisionLayer::CollisionLayerType)p_colObj0->getBroadphaseHandle()->m_collisionFilterGroup;
	CollisionLayer::CollisionLayerType obj0Msk = (CollisionLayer::CollisionLayerType)p_colObj0->getBroadphaseHandle()->m_collisionFilterMask;
	CollisionLayer::CollisionLayerType obj1Grp = (CollisionLayer::CollisionLayerType)p_colObj1->getBroadphaseHandle()->m_collisionFilterGroup;
	CollisionLayer::CollisionLayerType obj1Msk = (CollisionLayer::CollisionLayerType)p_colObj1->getBroadphaseHandle()->m_collisionFilterMask;
	bool collides = (obj0Grp & obj1Msk) != 0;
	collides = collides && (obj1Grp & obj0Msk);
	return collides;
}