#pragma once
#include <glm\gtc\type_ptr.hpp>
// =======================================================================================
//                                      IK2Handler
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Handler and solver for 2-link IK setup
///        
/// # IK2Handler
/// 
/// 11-7-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class IK2Handler
{
public:
	IK2Handler();
	~IK2Handler();


	void solve(const glm::vec3& p_footPosL, const glm::vec3& p_upperLegJointPosL, float p_upperLegLen, float p_lowerLegLen);

	float getUpperLegAngle() const;
	float getLowerLocalLegAngle() const;
	float getLowerWorldLegAngle() const;

	glm::vec3& getHipPosL();
	glm::vec3& getKneePosL();
	glm::vec3& getFootPosL();
protected:
private:
	void updateAngles(float p_lowerAngle, float p_upperAngle);
	float m_upperAngle;
	float m_lowerAngle;
	glm::vec3 m_hipPos;
	glm::vec3 m_kneePos;
	glm::vec3 m_endPos;
};