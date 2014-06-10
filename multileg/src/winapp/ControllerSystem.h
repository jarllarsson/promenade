#pragma once
#include <Artemis.h>
#include "TransformComponent.h"
#include "RigidBodyComponent.h"
#include <vector>
#include "ControllerComponent.h"

// =======================================================================================
//                                 ControllerSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	The specialized controller system that builds the controllers and
///			inits the kernels and gathers their results
///			This contains the run-time logic and data for the controllers.
///			The controller components themselves only contain structural data (info on how to handle the run-time data)
///        
/// # ControllerSystem
/// 
/// 20-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

#define MULTI

class ControllerSystem : public artemis::EntityProcessingSystem
{
private:
	struct VelocityStat
	{
		glm::vec3 m_oldPos;
		glm::vec3 m_currentVelocity;
		glm::vec3 m_desiredVelocity;
		glm::vec3 m_goalVelocity;
	};

	artemis::ComponentMapper<ControllerComponent> controllerComponentMapper;
	// Controller run-time data
	std::vector<ControllerComponent*> m_controllersToBuild;
	std::vector<ControllerComponent*> m_controllers;
	std::vector<VelocityStat>		  m_controllerVelocityStats;
	// Joint run-time data
	std::vector<glm::vec3>		m_jointTorques;
	std::vector<btRigidBody*>	m_jointRigidBodies;
	std::vector<glm::mat4>		m_jointWorldTransforms;
	std::vector<float>			m_jointLengths;
	std::vector<glm::vec4>		m_jointWorldEndpoints;
public:
	ControllerSystem()
	{
		addComponentType<ControllerComponent>();
		m_runTime = 0.0f;
		//addComponentType<RigidBodyComponent>();
		//m_dynamicsWorldPtr = p_dynamicsWorld;
	}

	virtual void initialize()
	{
		controllerComponentMapper.init(*world);
		//rigidBodyMapper.init(*world);
	}

	virtual void removed(artemis::Entity &e);

	virtual void added(artemis::Entity &e);

	virtual void processEntity(artemis::Entity &e);

	void update(float p_dt);

	void finish();

	void applyTorques();

	// Build uninited controllers, this has to be called 
	// after constraints & rb's have been inited by their systems
	void buildCheck();
private:
	// Control logic functions
	void controllerUpdate(int p_controllerId, float p_dt);
	void updateVelocityStats(int p_controllerId, ControllerComponent* p_controller, float p_dt);
	void updateFeet(int p_controllerId, ControllerComponent* p_controller);
	void updateTorques(int p_controllerId, ControllerComponent* p_controller, float p_dt);

	// Helper functions
	unsigned int addJoint(RigidBodyComponent* p_jointRigidBody, TransformComponent* p_jointTransform);
	void saveJointMatrix(unsigned int p_rigidBodyIdx);
	void saveJointWorldEndpoint(unsigned int p_idx, glm::mat4& p_worldMatPosRot);
	void initControllerVelocityStat(unsigned int p_idx);
	glm::vec3 getControllerPosition(unsigned int p_controllerId);
	glm::vec3 getControllerPosition(ControllerComponent* p_controller);
	glm::vec3 DOFAxisByVecCompId(unsigned int p_id);

	// global variables
	float m_runTime;
};
