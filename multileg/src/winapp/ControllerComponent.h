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
	unsigned int legFrameIdx;
	unsigned int leftHipIdx;
	unsigned int legSegments;

	std::vector<glm::vec3> legDOFChain;
	std::vector<unsigned int> jointIDXChain;

	// Specify entry points on construction, during build
	// the chains(lists) will be constructed by walking the pointer chain(double linked list)
	ControllerComponent(artemis::Entity* p_legFrame, artemis::Entity* p_upperLeg)
	{
		LegFrame legFrame{ p_legFrame, p_upperLeg };
		m_legFrames.push_back(legFrame);
	}

	virtual ~ControllerComponent();

	struct LegFrame
	{
		artemis::Entity* m_legFrameEntity;
		artemis::Entity* m_upperLegEntity;
	};
	std::vector<LegFrame> m_legFrames;

protected:
private:


};