#pragma once
#include <LinearMath/btScalar.h>
#include <vector>

class btDynamicsWorld;
class AdvancedEntitySystem;
class ControllerSystem;

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
	PhysicsWorldHandler(btDynamicsWorld* p_world, ControllerSystem* p_controllerSystem);

	void physProcessCallback(btScalar timeStep);

	unsigned int getNumberOfInternalSteps();

	void addPreprocessSystem(AdvancedEntitySystem* p_system);
	void addOrderIndependentSystem(AdvancedEntitySystem* p_system);

	void processPreprocessSystemCollection(float p_dt);
	void processOrderIndependentSystemCollection(float p_dt);

protected:
	// Might want to change this to generic list of a common base class
	ControllerSystem* m_controllerSystem; // But right now, we only need it for the controllers

	std::vector<AdvancedEntitySystem*> m_orderIndependentSystems;
	std::vector<AdvancedEntitySystem*> m_preprocessSystems;
	//
	// Physics world
	btDynamicsWorld* m_world;
	unsigned int m_internalStepCounter;
	void handleCollisions();
};

void physicsSimulationTickCallback(btDynamicsWorld *world, btScalar timeStep);