#pragma once

#include <Artemis.h>
#include "TransformComponent.h"
#include "RigidBodyComponent.h"
#include <btBulletDynamicsCommon.h>
#include "ConstraintComponent.h"
#include <vector>
#include <MeasurementBin.h>
#include <UniqueIndexList.h>
#include <Util.h>

// =======================================================================================
//                                      RigidBodySystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Updates transforms based on the result on rigidbodies
///        
/// # RigidBodySystem
/// 
/// 15-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class RigidBodySystem : public artemis::EntityProcessingSystem
{
private:
	artemis::ComponentMapper<TransformComponent> transformMapper;
	artemis::ComponentMapper<RigidBodyComponent> rigidBodyMapper;
	btDiscreteDynamicsWorld* m_dynamicsWorldPtr;
	// Vector to store creation calls for constraints
	// This is used so they can be inited in the correct order
	std::vector<artemis::Entity*> m_constraintCreationsList;
	// List for entities that have rigidbodies so they can be accessed by id
	UniqueIndexList<artemis::Entity*> m_rigidBodyEntities;
public:


	RigidBodySystem(btDiscreteDynamicsWorld* p_dynamicsWorld, MeasurementBin<string>* p_stateDbgRecorder=NULL) 
	{
		addComponentType<TransformComponent>();
		addComponentType<RigidBodyComponent>();
		m_dynamicsWorldPtr = p_dynamicsWorld;
		m_stateDbgRecorder = p_stateDbgRecorder;
	};

	virtual void initialize() 
	{
		transformMapper.init(*world);
		rigidBodyMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e);

	virtual void added(artemis::Entity &e);

	virtual void processEntity(artemis::Entity &e);

	// Void this has to be called explicitly for it to be done correctly
	// constraints need both its rigidbodies to have been added to the physics world
	// ie. after all entity adds. I can't control the order of adds, unlike processing.
	void executeDeferredConstraintInits();

	// Other deferred updates
	void lateUpdate();

private:
	void checkForNewConstraints(artemis::Entity &e);
	//void checkForConstraintsToRemove(artemis::Entity &e, RigidBodyComponent* p_rigidBody);
	void setupConstraints(artemis::Entity *e);
	MeasurementBin<string>* m_stateDbgRecorder;
	std::string m_stateString;
};
