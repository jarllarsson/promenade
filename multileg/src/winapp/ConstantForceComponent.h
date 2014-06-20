#pragma once
#include <EntityProcessingSystem.h>
#include <glm\gtc\type_ptr.hpp>
// =======================================================================================
//                                      ConstantForceComponent
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	For applying a force over a specified time or indefinitiely on a rigid body
///        
/// # ConstantForceComponent
/// 
/// 20-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ConstantForceComponent : public artemis::Component
{
public:
	ConstantForceComponent(glm::vec3 p_forceVec, float p_lifeTime = -1.0f)
	{
		if (p_lifeTime < 0.0f)
			m_timeBound = false;
		else
		{
			m_time = p_lifeTime;
			m_timeBound = true;
		}
		m_force = p_forceVec;
	}
	~ConstantForceComponent() {}

	glm::vec3 m_force;
	float m_time;
	bool m_timeBound;
protected:
private:
};