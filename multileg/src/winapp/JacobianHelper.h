#pragma once
#include "ControllerComponent.h"

// =======================================================================================
//                                      JacobianHelper
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Used to calculate Jacobian matrices for DOF-chains.
///			These are then used when applying the virtual forces over said chain as torques
///        
/// # JacobianHelper
/// 
/// 11-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

namespace JacobianHelper
{
	CMatrix calculateVFChainJacobian(const ControllerComponent::VFChain& p_chain, 
		const glm::vec3& p_currentChainEndpointGoalPos, 
		const std::vector<glm::vec3>* p_vfList,
		const std::vector<glm::vec4>* p_jointWorldAxes,
		const std::vector<glm::mat4>* p_jointWorldTransforms,
		unsigned int p_dofCount);
};