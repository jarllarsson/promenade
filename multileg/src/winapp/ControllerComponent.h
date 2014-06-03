#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"
#include <glm\gtc\matrix_transform.hpp>
#include "ConstraintComponent.h"
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
	// Temporary
	//unsigned int legFrameIdx;
	//unsigned int leftHipIdx;
	//unsigned int legSegments;

	struct Chain
	{
	public:
		std::vector<glm::vec3> DOFChain;
		std::vector<unsigned int> jointIDXChain;
		unsigned int getSize()
		{
			return DOFChain.size();
		}
	};

	// Chain constructs
	// The "ordinary" chain of legs, from leg frame to foot
	// Structure:
	// [R][1][2][F]
	Chain m_DOFChain;
	// Each link will all its DOFs to the chain
	// This will result in 0 to 3 vec3:s. (We're only using angles)

	// The gravity compensation chain, from start to foot, for each link in the chain
	// Structure construction:
	// [R][1][2][F] +
	//    [1][2][F] +
	//       [2][F] +
	//          [F] =
	// Structure:
	// [R][1][2][F][1][2][F][2][F][F]
	Chain m_DOFChainGravityComp;

	// Specify entry points on construction, during build
	// the chains(lists) will be constructed by walking the pointer chain(double linked list)
	ControllerComponent(artemis::Entity* p_legFrame, artemis::Entity* p_upperLeg)
	{
		LegFrame legFrame{ p_legFrame, p_upperLeg };
		m_legFrames.push_back(legFrame);
	}

	virtual ~ControllerComponent() {}

	struct LegFrame
	{
		artemis::Entity* m_legFrameEntity;
		artemis::Entity* m_upperLegEntity;
	};
	std::vector<LegFrame> m_legFrames;

protected:
private:


};