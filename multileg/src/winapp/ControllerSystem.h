#pragma once
#include <Artemis.h>
#include "TransformComponent.h"
#include "RigidBodyComponent.h"
#include <btBulletDynamicsCommon.h>
#include "ConstraintComponent.h"
#include <vector>
#include "ControllerComponent.h"
#include <ToString.h>
#include <DebugPrint.h>
// =======================================================================================
//                                 ControllerSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	The specialized controller system that builds the controllers and
///			inits the kernels and gathers their results
///        
/// # ControllerSystem
/// 
/// 20-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ControllerSystem : public artemis::EntityProcessingSystem
{
private:
	artemis::ComponentMapper<ControllerComponent> controllerComponentMapper;
	std::vector<ControllerComponent*> m_controllers;
	std::vector<glm::vec3> m_torques;
	std::vector<btRigidBody*> m_rigidBodies;
public:
	ControllerSystem()
	{
		addComponentType<ControllerComponent>();
		//addComponentType<RigidBodyComponent>();
		//m_dynamicsWorldPtr = p_dynamicsWorld;
	};

	virtual void initialize()
	{
		controllerComponentMapper.init(*world);
		//rigidBodyMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e);

	virtual void added(artemis::Entity &e);

	virtual void processEntity(artemis::Entity &e);

	void start(float p_dt);

	void finish();

	void applyTorques();
};

void ControllerSystem::removed(artemis::Entity &e)
{

}

void ControllerSystem::added(artemis::Entity &e)
{
	ControllerComponent* controller = controllerComponentMapper.get(e);
	m_controllers.push_back(controller);
}

void ControllerSystem::processEntity(artemis::Entity &e)
{

}

void ControllerSystem::start(float p_dt)
{
	DEBUGPRINT(( (std::string("\nController start DT=") + toString(p_dt) + "\n").c_str() ));
	for (int i = 0; i < m_torques.size(); i++)
	{
		m_torques[i] = glm::vec3(1.0f, 0.0f, 0.0f);
	}
}

void ControllerSystem::finish()
{

}

void ControllerSystem::applyTorques()
{
	if (m_rigidBodies.size()==m_torques.size())
	{
		for (int i = 0; i < m_rigidBodies.size(); i++)
		{
			glm::vec3* t = &m_torques[i];
			m_rigidBodies[i]->applyTorque(btVector3(t->x, t->y, t->z));
		}
	}
}
