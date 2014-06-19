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

class ControllerComponent : public artemis::Component
{
public:



	// Playback specific data and handlers
	// ==============================================================================
	GaitPlayer m_player;
	glm::vec3 m_goalVelocity;




	// Constructor and Destructor
	// ==============================================================================
	// Specify entry points on construction, during build
	// the chains(lists) will be constructed by walking the pointer chain(double linked list)
	ControllerComponent(artemis::Entity* p_legFrame, std::vector<artemis::Entity*>& p_hipJoints)
	{
		// for each inputted leg-frame entity...

		// Set up the entity-based leg frame representation
		// This is simply a struct of pointers to the artemis equivalents of
		// what the controller system will work with as joints and decomposed DOF-chains
		LegFrameEntityConstruct legFrameEntityConstruct;
		legFrameEntityConstruct.m_legFrameEntity = p_legFrame;
		// Add all legs to it
		for (unsigned int i = 0; i < p_hipJoints.size();i++)
		{
			legFrameEntityConstruct.m_upperLegEntities.push_back(p_hipJoints[i]);
		}			
		// add to our list of constructs
		m_legFrameEntityConstructs.push_back(legFrameEntityConstruct);
		// Create the leg frame data struct as well
		// Allocate it according to number of leg entities that was inputted
		LegFrame legFrame;
		legFrame.m_stepCycles.resize(legFrameEntityConstruct.m_upperLegEntities.size());
		legFrame.m_stepCycles[1].m_tuneStepTrigger = 0.5f;
		m_legFrames.push_back(legFrame);
	}

	virtual ~ControllerComponent() {}
	


	// Internal data types
	// ==============================================================================
	// Virtual force chain
	struct VFChain
	{
	public:
		std::vector<glm::vec3> DOFChain;
		std::vector<unsigned int> jointIDXChain;
		// VF vector here maybe?
		glm::vec3 vf;
		unsigned int getSize() const
		{
			return (unsigned int)DOFChain.size();
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
	struct LegFrame
	{
		std::vector<StepCycle> m_stepCycles;
		unsigned int m_legFrameJointId;
		unsigned int m_spineJointId;
		std::vector<Leg> m_legs;
		std::vector<unsigned int> m_feetJointId;
		std::vector<unsigned int> m_hipJointId;
	};

	// Construction description struct for leg frames
	// This is the artemis based input
	struct LegFrameEntityConstruct
	{
		artemis::Entity* m_legFrameEntity;
		std::vector<artemis::Entity*> m_upperLegEntities;
	};

	// Leg frame lists access and handling
	const unsigned int getLegFrameCount() const {return (unsigned int)m_legFrames.size();}
	LegFrame* getLegFrame(unsigned int p_idx)								{return &m_legFrames[p_idx];}
	LegFrameEntityConstruct* getLegFrameEntityConstruct(unsigned int p_idx) {return &m_legFrameEntityConstructs[p_idx];}
	// ==============================================================================

	// Torque list access and handling
	const unsigned int getTorqueListOffset() const { return m_torqueListOffset; }
	const unsigned int getTorqueListChunkSize() const { return m_torqueListChunkSize; }
	void setTorqueListProperties(unsigned int p_offset, unsigned int p_size) { m_torqueListOffset = p_offset; m_torqueListChunkSize = p_size; }

protected:
private:
	// Leg frame lists
	// ==============================================================================
	std::vector<LegFrame> m_legFrames;
	std::vector<LegFrameEntityConstruct> m_legFrameEntityConstructs;

	// Torque list nav
	unsigned int m_torqueListOffset;
	unsigned int m_torqueListChunkSize;
};


