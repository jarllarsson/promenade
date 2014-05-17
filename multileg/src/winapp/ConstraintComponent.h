#pragma once
#include <Artemis.h>
#include <btBulletDynamicsCommon.h>
#include <glm\gtc\type_ptr.hpp>
#include "TransformComponent.h"
#include <glm\gtc\matrix_transform.hpp>
#include "RigidBodyComponent.h"
// =======================================================================================
//                                      ConstraintComponent
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # ConstraintComponent
/// 
/// 17-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ConstraintComponent : public artemis::Component
{
public:
	struct ConstraintInit
	{
		glm::vec3 m_localAxis;
		glm::vec3 m_parentLocalAxis;
		glm::vec3 m_localAnchor;
		glm::vec3 m_parentLocalAnchor;
		/*
		For each axis, if
		lower limit = upper limit
			The axis is locked
		lower limit < upper limit
			The axis is limited between the specified values
		lower limit > upper limit
			The axis is free and has no limits
		*/
		glm::vec3 m_linearDOF_LULimits[2];
		glm::vec3 m_angularDOF_LULimits[2];
	};


	ConstraintComponent(RigidBodyComponent* p_otherBody, const ConstraintInit& p_desc)
	{
		m_otherBody = p_otherBody;
	}

	virtual ~ConstraintComponent()
	{
		delete m_constraint;
	}
private:
	RigidBodyComponent* m_otherBody;
	btGeneric6DofConstraint* m_constraint;
};