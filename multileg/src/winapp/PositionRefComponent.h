#pragma once
#include <Artemis.h>
#include <glm\gtc\type_ptr.hpp>
// =======================================================================================
//                                      PositionRefComponent
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Keeps a reference to a glm::vec3. Is used to update transforms with this vector.
///			For debugging purposes
///        
/// # PositionRefComponent
/// 
/// 23-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class PositionRefComponent : public artemis::Component
{
public:
	PositionRefComponent(glm::vec3* p_posRef)
	{
		m_posRef = p_posRef;
	}
	virtual ~PositionRefComponent() {}


	glm::vec3* m_posRef;
protected:
private:
};