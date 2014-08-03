#pragma once
#include <Artemis.h>
#include "TransformComponent.h"
#include "RigidBodyComponent.h"
#include <vector>
#include "ControllerComponent.h"
#include "AdvancedEntitySystem.h"
#include <MeasurementBin.h>

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
public:
	// Used to control and read velocity specifics per controller
	struct VelocityStat
	{
		VelocityStat(const glm::vec3& p_pos, const glm::vec3& p_goalGaitVelocity)
		{
			m_oldPos = p_pos;
			m_currentVelocity=glm::vec3(0.0f);
			m_desiredVelocity=glm::vec3(0.0f);
			m_goalVelocity = p_goalGaitVelocity;
		}
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
private:
	artemis::ComponentMapper<ControllerComponent> controllerComponentMapper;
	// Controller run-time data
	std::vector<ControllerComponent*> m_controllersToBuild;
	std::vector<ControllerComponent*> m_controllers;
	std::vector<VelocityStat>		  m_controllerVelocityStats;
	std::vector<LocationStat>		  m_controllerLocationStats;
	// VF run-time data
	std::vector<glm::vec3>		m_VFs;
	// Joint run-time data
	std::vector<glm::vec3>		m_jointTorques;
	std::vector<btRigidBody*>	m_jointRigidBodies;
	std::vector<RigidBodyComponent*> m_rigidBodyRefs;
	std::vector<glm::mat4>		m_jointWorldTransforms;
	std::vector<float>			m_jointLengths;
	std::vector<float>			m_jointMass;
	std::vector<glm::vec4>		m_jointWorldInnerEndpoints;
	std::vector<glm::vec4>		m_jointWorldOuterEndpoints;
	// Other joint run time data, for debugging
	std::vector<artemis::Entity*>	m_dbgJointEntities;
public:
	ControllerSystem(MeasurementBin<float>* p_perfMeasurer=NULL)
	{
		addComponentType<ControllerComponent>();
		m_runTime = 0.0f;
		m_steps = 0;
		//addComponentType<RigidBodyComponent>();
		//m_dynamicsWorldPtr = p_dynamicsWorld;
		m_useVFTorque = true;
		m_useCGVFTorque = true;
		m_usePDTorque = true;
		m_perfRecorder = p_perfMeasurer;
		m_timing = 0;
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

	virtual void fixedUpdate(float p_dt);

	void finish();

	void applyTorques(float p_dt);

	// Build uninited controllers, this has to be called 
	// after constraints & rb's have been inited by their systems
	void buildCheck();

	// Public helper functions
	glm::mat4& getLegFrameTransform(const ControllerComponent::LegFrame* p_lf);
	VelocityStat& getControllerVelocityStat(const ControllerComponent* p_controller);
	glm::vec3 getJointAcceleration(unsigned int p_jointId);
	double getLatestTiming();

private:
	// build helpers
	// Add a joint's all DOFs to chain
	void addJointToStandardVFChain(ControllerComponent::VFChain* p_legVFChain, unsigned int p_idx, 
		unsigned int p_vfIdx, const glm::vec3* p_angularLims = NULL);
	void addJointToPDChain(ControllerComponent::PDChain* p_legChain, unsigned int p_idx, float p_kp, float p_kd);
	// Add chain DOFs to list again, from Joint-offset ( this functions skips the appropriate number of DOFs)
	void repeatAppendChainPart(ControllerComponent::VFChain* p_legVFChain, 
		int p_localJointOffset, int p_jointCount, unsigned int p_originalChainSize);

	// Control logic functions
	void controllerUpdate(unsigned int p_controllerId, float p_dt);
	void updateLocationAndVelocityStats(int p_controllerId, ControllerComponent* p_controller, float p_dt);
	void updateFeet(unsigned int p_controllerId, ControllerComponent* p_controller);
	void updateTorques(unsigned int p_controllerId, ControllerComponent* p_controller, float p_dt);

	// Leg frame logic functions
	void calculateLegFrameNetLegVF(unsigned int p_controllerIdx, ControllerComponent::LegFrame* p_lf, float p_phi, float p_dt, VelocityStat& p_velocityStats);

	// Leg logic functions
	bool isInControlledStance(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi);
	// Forces
	glm::vec3 calculateFsw(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi, float p_dt);
	glm::vec3 calculateFv(ControllerComponent::LegFrame* p_lf, const VelocityStat& p_velocityStats);
	glm::vec3 calculateFh(ControllerComponent::LegFrame* p_lf, const LocationStat& p_locationStat, float p_phi, float p_dt, const glm::vec3& p_up);
	glm::vec3 calculateFd(unsigned int p_controllerIdx, ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx);
	// Virtual force and VF-to-torque algorithms
	glm::vec3 calculateSwingLegVF(const glm::vec3& p_fsw);
	glm::vec3 calculateStanceLegVF(unsigned int p_stanceLegCount,
		const glm::vec3& p_fv, const glm::vec3& p_fh, const glm::vec3& p_fd);
	void computeAllVFTorques(std::vector<glm::vec3>* p_outTVF, 
		ControllerComponent* p_controller, unsigned int p_controllerIdx, 
		unsigned int p_torqueIdxOffset, 
		float p_phi, float p_dt);
	void computeVFTorquesFromChain(std::vector<glm::vec3>* p_outTVF, 
		ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx,
		ControllerComponent::VFChainType p_type, unsigned int p_torqueIdxOffset, 
		float p_phi, float p_dt);
	void applyNetLegFrameTorque(unsigned int p_controllerId, ControllerComponent* p_controller, unsigned int p_legFrameIdx, float p_phi, float p_dt);
	// PD calculation for legs
	void computePDTorques(std::vector<glm::vec3>* p_outTVF, 
		ControllerComponent* p_controller, unsigned int p_controllerIdx, 
		unsigned int p_torqueIdxOffset, 
		float p_phi, float p_dt);

	// Foot placement model	
	glm::vec3 getFootPos(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx);	
	void updateFoot(unsigned int p_controllerId, ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi, const glm::vec3& p_velocity, const glm::vec3& p_desiredVelocity, const glm::vec3& p_groundPos);
	void updateFootStrikePosition(unsigned int p_controllerId, ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi, const glm::vec3& p_velocity, const glm::vec3& p_desiredVelocity, const glm::vec3& p_groundPos);
	void updateFootSwingPosition(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi);
	void offsetFootTargetDownOnLateStrike(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx);
	glm::vec3 projectFootPosToGround(const glm::vec3& p_footPosLF, const glm::vec3& p_groundPos) const;
	glm::vec3 calculateVelocityScaledFootPos(const ControllerComponent::LegFrame* p_lf, const glm::vec3& p_footPosLF, const glm::vec3& p_velocity, const glm::vec3& p_desiredVelocity) const;
	float getFootTransitionPhase(ControllerComponent::LegFrame* p_lf, float p_swingPhi);

	// Internal helper functions
	unsigned int addJoint(RigidBodyComponent* p_jointRigidBody, TransformComponent* p_jointTransform);
	void saveJointMatrix(unsigned int p_rigidBodyIdx);
	void saveJointWorldEndpoints(unsigned int p_idx, glm::mat4& p_worldMatPosRot);
	void initControllerLocationAndVelocityStat(unsigned int p_idx, const glm::vec3& p_gaitGoalVelocity);
	glm::vec3 getControllerPosition(unsigned int p_controllerId);
	glm::vec3 getControllerPosition(ControllerComponent* p_controller);
	glm::vec3 getLegFramePosition(const ControllerComponent::LegFrame* p_lf) const;
	glm::vec3 DOFAxisByVecCompId(unsigned int p_id);
	glm::mat4 getDesiredWorldOrientation(unsigned int p_controllerId) const;
	bool isFootStrike(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx);
	void writeFeetCollisionStatus(ControllerComponent* p_controller);
	float getDesiredFootAngle(unsigned int p_legIdx, ControllerComponent::LegFrame* p_lf, float p_phi);
	// global variables
	float m_runTime;
	int m_steps;
	bool m_useVFTorque;
	bool m_useCGVFTorque;
	bool m_usePDTorque;

	// Dbg
	MeasurementBin<float>* m_perfRecorder;
	double m_timing;
};
