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
	ControllerComponent();
	virtual ~ControllerComponent();
protected:
private:

};