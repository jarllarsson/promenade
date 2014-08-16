#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"
#include <glm\gtc\matrix_transform.hpp>
#include "ConstraintComponent.h"
#include "GaitPlayer.h"
#include "StepCycle.h"
#include <vector>
#include "PieceWiseLinear.h"
#include "PDn.h"
#include "PD.h"
#include "IK2Handler.h"
#include <IOptimizable.h>

// =======================================================================================
//                                      ControllerComponent
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Component that defines the structure of a character
///			
///        
/// # ControllerComponent
/// 
/// 20-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ControllerComponent : public artemis::Component, public IOptimizable
{
public:



	// Playback specific data and handlers
	// ==============================================================================
	GaitPlayer m_player;
	glm::vec3 m_goalVelocity;


	enum Orientation{ YAW = 0, PITCH = 1, ROLL = 2 };

	// Constructor and Destructor
	// ==============================================================================
	// Specify entry points on construction, during build
	// the chains(lists) will be constructed by walking the pointer chain(double linked list)
	ControllerComponent(artemis::Entity* p_legFrame, std::vector<artemis::Entity*>& p_hipJoints);

	virtual ~ControllerComponent() {}
	


	// Internal data types
	// ==============================================================================
	// Virtual force chain, for legs
	struct VFChain
	{
	public:
		std::vector<glm::vec3> m_DOFChain;
		std::vector<unsigned int> m_jointIdxChain;
		// vector with indices to global virtual force list
		std::vector<unsigned int> m_vfIdxList;
		//
		unsigned int getSize() const
		{
			return (unsigned int)m_DOFChain.size();
		}
	};
	enum VFChainType
	{
		STANDARD_CHAIN,
		GRAVITY_COMPENSATION_CHAIN
	};

	// PD driver chain, for legs
	struct PDChain
	{
	public:
		std::vector<PDn> m_PDChain;
		std::vector<unsigned int> m_jointIdxChain;

		unsigned int getSize() const
		{
			return (unsigned int)m_PDChain.size();
		}

		unsigned int getUpperLegSegmentIdx() const
		{
			return 0; // NOTE! !HACK! HARDCODED!
		}

		unsigned int getLowerLegSegmentIdx() const
		{
			return 1; // NOTE! !HACK! HARDCODED!
		}

		unsigned int getFootIdx() const
		{
			return 2;
				//(unsigned int)m_PDChain.size()-1; // NOTE! !HACK! HARDCODED!
		}

		unsigned int getUpperJointIdx() const
		{
			return m_jointIdxChain[getUpperLegSegmentIdx()];
		}

		unsigned int getLowerJointIdx() const
		{
			return m_jointIdxChain[getLowerLegSegmentIdx()];
		}

		unsigned int getFootJointIdx() const
		{
			return m_jointIdxChain[getFootIdx()];
		}
	};

	// Leg
	// Contains information for per-leg actions
	struct Leg
	{
		// Chain constructs
		// ==============================================================================
		// Each link will all its DOFs to the chain
		// This will result in 0 to 3 vec3:s. (We're only using angles)
		// ==============================================================================

		// The "ordinary" chain of legs, from leg frame to foot
		// Structure:
		// [R][1][2][F]
		VFChain m_DOFChain;

		// The gravity compensation chain, from start to foot, for each link in the chain
		// Structure construction:
		// [R][1][2][F] +
		//    [1][2][F] +
		//       [2][F] +
		//          [F] =
		// Structure:
		// [R][1][2][F][1][2][F][2][F][F]
		VFChain m_DOFChainGravityComp;

		// ==============================================================================

		// PD chain for keeping internal segment orientation
		// Knee position(upp angle+lower angle) is ik-based and foot angle
		// is based on an optimizable trajectory
		PDChain m_PDChain;

		// ==============================================================================
		VFChain* getVFChain(VFChainType p_type)
		{
			VFChain* res = &m_DOFChain;
			if (p_type == VFChainType::STANDARD_CHAIN)
				res=&m_DOFChain;
			else
				res=&m_DOFChainGravityComp;
			return res;
		}

		PDChain* getPDChain()
		{
			return &m_PDChain;
		}
	};

	// Leg frame
	// ==============================================================================
	// As our systems will interface with artemis, the leg frame structure has been
	// split into a run-time specific structure for the handling of locomotion
	// and an entity specific structure for handling the artemis interop.
	// These are put into two separate lists of the same size, where each entry mirror
	// the other.
	// ==============================================================================
	// Run-time leg frame structure
	struct LegFrame : public IOptimizable
	{
		LegFrame()
		{
			// Trajectory settings
			m_orientationLFTraj[(unsigned int)Orientation::PITCH].reset(PieceWiseLinear::FLAT,TWOPI); // try to stay upside down
			m_heightLFTraj.reset(PieceWiseLinear::FULL, 10.0f); // is reinited to character height in build
			m_stepHeighTraj.reset(PieceWiseLinear::COS_INV_NORM, 1.0f); // Stepping is defaulted to an arc
			m_footTrackingGainKp.reset(PieceWiseLinear::LIN_INC,1.0f); // Foot tracking controller for fast gaits. linear(=t) by default
			m_footTransitionEase.reset(PieceWiseLinear::LIN_INC,1.0f); // Easing on sagittal movement is linear(=t) by default	
			// PD settings
			m_desiredLFTorquePD.setKp_KdEQTenPrcntKp(30.0f);
			m_FhPD.setKp_KdEQTenPrcntKp(3.0f);
			m_footTrackingSpringDamper.setKp_KdEQTenPrcntKp(1.0f);
			// Vectors and Floats
			m_stepLength = glm::vec2(0.5f, 1.0f);
			m_footPlacementVelocityScale = 1.0f;
			m_height = 0.0f;
			m_lateStrikeOffsetDeltaH = 0.0f;
			m_velocityRegulatorKv = 3.0f;
			m_FDHVComponents = glm::vec4(0.0f);
				//glm::vec4(-0.1f, 0.2f, 0.0f, 0.1f);
			//
			m_legPDsKp = 3.0f;
			m_legPDsKd = 0.3f;
			// foot
			m_tuneToeOffAngle=HALFPI;				
			m_tuneFootStrikeAngle = -HALFPI/9.0f;
		}

		// Structure ids
		unsigned int m_legFrameJointId;				// per leg frame
		int m_spineJointId;				// per leg frame
		std::vector<unsigned int> m_feetJointId;	// per leg
		std::vector<unsigned int> m_hipJointId;		// per leg		
		std::vector<unsigned int> m_footRigidBodyIdx;	// Idx to foot rigidbody in foot list in system for special collision check, per leg	
		// Playback data
		std::vector<StepCycle> m_stepCycles;			// per leg	
		PieceWiseLinear		   m_orientationLFTraj[3];	// xyz-orientation trajectory, per leg frame
		PieceWiseLinear		   m_heightLFTraj;			// height trajectory, per leg frame
		PieceWiseLinear		   m_footTrackingGainKp;	// Variable proportionate gain for swing phase, per leg frame
		PieceWiseLinear		   m_stepHeighTraj;			// stepping height trajectory(for legs/feet), per leg frame
		PieceWiseLinear		   m_footTransitionEase;	// Easing function for swing speed(for legs/feet), per leg frame
		PDn					   m_desiredLFTorquePD;		// goal torque, per leg frame
		PD					   m_FhPD;					// driver used to try to reach the desired height VF (Fh)
		PD					   m_footTrackingSpringDamper;
		float				   m_lateStrikeOffsetDeltaH;	// Offset used as punishment on foot placement y on late strike. per leg frame
		float				   m_velocityRegulatorKv;		// Gain for regulating velocity
		glm::vec4			   m_FDHVComponents;		// Components to FD lin func, vertical and horizontal in sagittal plane
		std::vector<IK2Handler> m_legIK;				// IK handlers for legs

		// Special for now, define PD gains in these and they'll be used for all segment PDs in build:
		float			   m_legPDsKp;
		float			   m_legPDsKd;
		// Structure
		std::vector<Leg> m_legs;								// per leg
		std::vector<glm::vec3>  m_footStrikePlacement;			// The place on the ground where the foot should strike next, per leg
		std::vector<glm::vec3>	m_footLiftPlacement;			// From where the foot was lifted, per leg
		std::vector<bool>		m_footLiftPlacementPerformed;	// If foot just took off (and the "old" pos should be updated), per leg
		std::vector<glm::vec3>	m_footTarget;					// The current position in the foot's swing trajectory, per leg
		std::vector<bool>		m_footIsColliding;				// If foot is colliding, per leg
		
		// Foot rotation
		float m_tuneToeOffAngle;								// per leg frame
		std::vector<float> m_toeOffTime;					// "delta-t" for toe off, per leg
		float m_tuneFootStrikeAngle;							// per leg frame
		std::vector<float> m_tuneFootStrikeTime;   // "delta-t" for foot strike, per leg


		float			 m_footPlacementVelocityScale;			// per leg frame
		float			 m_height;								// per leg frame (max height, lf to feet)
		// NOTE!
		// I've lessened the amount of parameters by letting each leg in a leg frame share
		// per-leg parameters. Effectively mirroring behaviour over the saggital plane.
		// There are still two step cycles though, to allow for phase shifting.
		glm::vec2 m_stepLength;						// PLF, the coronal(x) and saggital(y) step distance. per leg frame

		// =============================================================
		// Access methods
		// =============================================================
		// Retrieves the current orientation quaternion from the
		// trajectory function at time phi.
		glm::quat getCurrentDesiredOrientation(float p_phi);

		// Drives the PD-controller and retrieves the 3-axis torque
		// vector that will be used as the desired torque for which the
		// stance legs tries to accomplish.
		glm::vec3 getOrientationPDTorque(const glm::quat& p_currentOrientation, 
							const glm::quat& p_desiredOrientation, float p_dt);

		// Helper function to correctly init the foot placement variables
		void createFootPlacementModelVarsForNewLeg(const glm::vec3& p_startPos);

		// Optimization
		virtual std::vector<float> getParams();
		virtual void consumeParams(std::vector<float>& p_other);
		virtual std::vector<float> getParamsMax();
		virtual std::vector<float> getParamsMin();
	};

	// Construction description struct for leg frames
	// This is the artemis based input
	struct LegFrameEntityConstruct
	{
		artemis::Entity* m_legFrameEntity;
		std::vector<artemis::Entity*> m_upperLegEntities;
	};

	unsigned int m_sysIdx;

	// Leg frame lists access and handling
	const unsigned int getLegFrameCount() const {return (unsigned int)m_legFrames.size();}
	LegFrame* getLegFrame(unsigned int p_idx)								{return &m_legFrames[p_idx];}
	LegFrameEntityConstruct* getLegFrameEntityConstruct(unsigned int p_idx) {return &m_legFrameEntityConstructs[p_idx];}
	void setToBuildComplete() { m_buildComplete = true; }
	bool isBuildComplete() { return m_buildComplete; }
	// ==============================================================================

	// Torque list access and handling
	const unsigned int getTorqueListOffset() const { return m_torqueListOffset; }
	const unsigned int getTorqueListChunkSize() const { return m_torqueListChunkSize; }
	void setTorqueListProperties(unsigned int p_offset, unsigned int p_size) { m_torqueListOffset = p_offset; m_torqueListChunkSize = p_size; }

	// Optimization
	virtual std::vector<float> getParams();
	virtual void consumeParams(std::vector<float>& p_other);
	virtual std::vector<float> getParamsMax();
	virtual std::vector<float> getParamsMin();
	unsigned int getHeadJointId();

protected:
private:
	bool m_buildComplete;

	// Leg frame lists
	// ==============================================================================
	std::vector<LegFrame> m_legFrames;
	std::vector<LegFrameEntityConstruct> m_legFrameEntityConstructs;

	// Torque list nav
	unsigned int m_torqueListOffset;
	unsigned int m_torqueListChunkSize;
};


