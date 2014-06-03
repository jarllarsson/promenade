#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"
#include <glm\gtc\matrix_transform.hpp>
#include "ConstraintComponent.h"
#include "GaitPlayer.h"
#include "StepCycle.h"
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
	// Internal data types
	// ==============================================================================
	struct Chain
	{
	public:
		std::vector<glm::vec3> DOFChain;
		std::vector<unsigned int> jointIDXChain;
		// VF vector here maybe?
		unsigned int getSize()
		{
			return (unsigned int)DOFChain.size();
		}
	};


	// Playback data and handler
	// ==============================================================================
	GaitPlayer m_player;


	// Chain constructs
	// ==============================================================================
	// Each link will all its DOFs to the chain
	// This will result in 0 to 3 vec3:s. (We're only using angles)
	// ==============================================================================

	// The "ordinary" chain of legs, from leg frame to foot
	// Structure:
	// [R][1][2][F]
	Chain m_DOFChain;

	// The gravity compensation chain, from start to foot, for each link in the chain
	// Structure construction:
	// [R][1][2][F] +
	//    [1][2][F] +
	//       [2][F] +
	//          [F] =
	// Structure:
	// [R][1][2][F][1][2][F][2][F][F]
	Chain m_DOFChainGravityComp;

	// ==============================================================================


	// Constructor and Destructor
	// ==============================================================================
	// Specify entry points on construction, during build
	// the chains(lists) will be constructed by walking the pointer chain(double linked list)
	ControllerComponent(artemis::Entity* p_legFrame, artemis::Entity* p_upperLeg)
	{		
		// for each inputted leg-frame entity...

		// Set up the entity-based leg frame representation
		// This is simply a struct of pointers to the artemis equivalents of
		// what the controller system will work with as joints and decomposed DOF-chains
		LegFrameEntityConstruct legFrameEntityConstruct;
		legFrameEntityConstruct.m_legFrameEntity=p_legFrame;
		// Add all legs to it
		legFrameEntityConstruct.m_upperLegEntities.push_back(p_upperLeg);
		// add to our list of constructs
		m_legFrameEntityConstructs.push_back(legFrameEntityConstruct);
		// Create the leg frame data struct as well
		// Allocate it according to number of leg entities that was inputted
		LegFrame legFrame;
		legFrame.m_stepCycles.resize(legFrameEntityConstruct.m_upperLegEntities.size());
		m_legFrames.push_back(legFrame);
	}

	virtual ~ControllerComponent() {}


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

protected:
private:
	// Leg frame lists
	// ==============================================================================
	std::vector<LegFrame> m_legFrames;
	std::vector<LegFrameEntityConstruct> m_legFrameEntityConstructs;

};