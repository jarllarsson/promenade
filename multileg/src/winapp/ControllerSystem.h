#pragma once
#include <Artemis.h>
#include "TransformComponent.h"
#include "RigidBodyComponent.h"
#include <vector>
#include "ControllerComponent.h"
#include "AdvancedEntitySystem.h"

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

//#define MULTI


class ControllerSystem : public AdvancedEntitySystem
{
private:
	// Used to control and read velocity specifics per controller
	struct VelocityStat
	{
		glm::vec3 m_oldPos;
		glm::vec3 m_currentVelocity;
		glm::vec3 m_desiredVelocity;
		glm::vec3 m_goalVelocity;
	};

	// Used to store and read location specifics per controller
	struct LocationStat
	{
		glm::vec3 m_worldPos;
		glm::vec3 m_currentGroundPos;
	};

	artemis::ComponentMapper<ControllerComponent> controllerComponentMapper;
	// Controller run-time data
	std::vector<ControllerComponent*> m_controllersToBuild;
	std::vector<ControllerComponent*> m_controllers;
	std::vector<VelocityStat>		  m_controllerVelocityStats;
	std::vector<LocationStat>		  m_controllerLocationStats;
	// Joint run-time data
	std::vector<glm::vec3>		m_jointTorques;
	std::vector<btRigidBody*>	m_jointRigidBodies;
	std::vector<glm::mat4>		m_jointWorldTransforms;
	std::vector<float>			m_jointLengths;
	std::vector<glm::vec4>		m_jointWorldInnerEndpoints;
	std::vector<glm::vec4>		m_jointWorldOuterEndpoints;
public:
	ControllerSystem()
	{
		addComponentType<ControllerComponent>();
		m_runTime = 0.0f;
		m_steps = 0;
		//addComponentType<RigidBodyComponent>();
		//m_dynamicsWorldPtr = p_dynamicsWorld;
		m_useVFTorque = true;
		m_useCGVFTorque = true;
		m_usePDTorque = true;
	}

	virtual ~ControllerSystem();

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

	void applyTorques(float p_dt);

	// Build uninited controllers, this has to be called 
	// after constraints & rb's have been inited by their systems
	void buildCheck();

	void addJointToChain(ControllerComponent::Leg* p_leg, unsigned int p_idx, const glm::vec3* p_angularLims=NULL);

private:
	// Control logic functions
	void controllerUpdate(int p_controllerId, float p_dt);
	void updateLocationAndVelocityStats(int p_controllerId, ControllerComponent* p_controller, float p_dt);
	void updateFeet(int p_controllerId, ControllerComponent* p_controller);
	void updateTorques(int p_controllerId, ControllerComponent* p_controller, float p_dt);

	// Leg frame logic functions
	void calculateLegFrameNetLegVF(unsigned int p_controllerIdx, ControllerComponent::LegFrame* p_lf, float p_phi, float p_dt, VelocityStat& p_velocityStats);

	// Leg logic functions
	bool isInControlledStance(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi);
	glm::vec3 calculateFsw(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi, float p_dt);
	glm::vec3 calculateFv(ControllerComponent::LegFrame* p_lf, const VelocityStat& p_velocityStats);
	glm::vec3 calculateFh(ControllerComponent::LegFrame* p_lf, const LocationStat& p_locationStat, float p_phi, float p_dt, const glm::vec3& p_up);
	glm::vec3 calculateFd(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx);
	glm::vec3 calculateSwingLegVF(const glm::vec3& p_fsw);
	glm::vec3 calculateStanceLegVF(unsigned int p_stanceLegCount,
		const glm::vec3& p_fv, const glm::vec3& p_fh, const glm::vec3& p_fd);

	// Helper functions
	unsigned int addJoint(RigidBodyComponent* p_jointRigidBody, TransformComponent* p_jointTransform);
	void saveJointMatrix(unsigned int p_rigidBodyIdx);
	void saveJointWorldEndpoints(unsigned int p_idx, glm::mat4& p_worldMatPosRot);
	void initControllerLocationAndVelocityStat(unsigned int p_idx);
	glm::vec3 getControllerPosition(unsigned int p_controllerId);
	glm::vec3 getControllerPosition(ControllerComponent* p_controller);
	glm::vec3 DOFAxisByVecCompId(unsigned int p_id);
	void computeVFTorques(std::vector<glm::vec3>* p_outTVF, ControllerComponent* p_controller, unsigned int p_controllerIdx, float p_phi, float p_dt);
	void applyNetLegFrameTorque(int p_controllerId, ControllerComponent* p_controller, unsigned int p_legFrameIdx, std::vector<glm::vec3>* m_jointTorques, unsigned int p_torqueIdxOffset, unsigned int p_torqueCount, float p_phi);
	// global variables
	float m_runTime;
	int m_steps;
	bool m_useVFTorque;
	bool m_useCGVFTorque;
	bool m_usePDTorque;
};
