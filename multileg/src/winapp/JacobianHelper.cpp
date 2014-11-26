#include "JacobianHelper.h"

CMatrix JacobianHelper::calculateVFChainJacobian(const ControllerComponent::VFChain& p_chain, 
												 const glm::vec3& p_currentChainEndpointGoalPos, 
												 const std::vector<glm::vec3>* p_vfList,
												 const std::vector<glm::vec4>* p_jointWorldAxes,
												 const std::vector<glm::mat4>* p_jointWorldTransforms,
												 unsigned int p_dofCount)
{
	// calculate size
	// If a separate root object was specified
	// we need to account for that extra joint
	unsigned int dofCount = p_dofCount;
	// Prepare Jacobian matrix
	CMatrix J(3, dofCount); // 3 is position in xyz
	for (unsigned int i = 0; i < dofCount; i++) // this is then the "thread pool"
	{
		// Fetch the id for the joint from the list
		unsigned int jointIdx = p_chain.m_jointIdxChain[i];
		// Start calculating the jacobian for the current DOF
		glm::vec3 jointAxisPos = MathHelp::toVec3((*p_jointWorldAxes)[jointIdx]);
		glm::vec3 vf = (*p_vfList)[p_chain.m_vfIdxList[i]];
		glm::vec3 dir = p_currentChainEndpointGoalPos + vf - jointAxisPos;
		//Debug.Log(linkPos.ToString());
		const glm::vec3* dof = &p_chain.m_DOFChain[i];
		// Solve for given axis
		glm::vec3 rotAxis = MathHelp::transformDirection((*p_jointWorldTransforms)[jointIdx], *dof);
		glm::vec3 dirTarget = glm::cross(rotAxis, dir);
		// Add result to matrix
		J(0, i) = dirTarget.x;
		J(1, i) = dirTarget.y;
		J(2, i) = dirTarget.z;
	}
	return J;
}
