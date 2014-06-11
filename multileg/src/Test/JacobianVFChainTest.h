#pragma once
/*#include <glm\core\type_vec3.hpp>
#include <vector>

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

CMatrix calculateVFChainJacobian(const VFChain& p_chain,
	const glm::vec3& p_currentChainEndpointGoalPos,
	const std::vector<glm::vec4>* p_jointWorldAxes,
	const std::vector<glm::mat4>* p_jointWorldTransforms)
{
	// calculate size
	// If a separate root object was specified
	// we need to account for that extra joint
	unsigned int dofCount = p_chain.getSize();
	// Prepare Jacobian matrix
	CMatrix J = CMatrix(3, dofCount); // 3 is position in xyz
	for (int i = 0; i < dofCount; i++) // this is then the "thread pool"
	{
		// Fetch the id for the joint from the list
		unsigned int jointIdx = p_chain.jointIDXChain[i];
		// Start calculating the jacobian for the current DOF
		glm::vec3 jointAxisPos = MathHelp::toVec3((*p_jointWorldAxes)[jointIdx]);
		//Debug.Log(linkPos.ToString());
		const glm::vec3* dof = &p_chain.DOFChain[i];
		// Solve for given axis
		glm::vec3 rotAxis = MathHelp::transformDirection((*p_jointWorldTransforms)[jointIdx], *dof);
		glm::vec3 dirTarget = glm::cross(rotAxis, p_currentChainEndpointGoalPos - jointAxisPos);
		// Add result to matrix
		J(0, i) = dirTarget.x;
		J(1, i) = dirTarget.y;
		J(2, i) = dirTarget.z;
	}
	return J;
}


TEST_CASE("Jacobian calculation", "[Jacobian,result]") 
{
	
	VFChain testChain;

	glm::vec3 vf = glm::vec3(0.0f, 100.0f, 0.0f);
	std::vector<glm::vec4> p_jointWorldAxes


	std::vector<glm::mat4> p_jointWorldTransforms

	SECTION("Assure same result with different world matrices, but same change distance") 
	{
		CMatrix mat(100, 10);
		REQUIRE(mat.m_rows == 100);
		REQUIRE(mat.m_cols == 10);
	}
}*/