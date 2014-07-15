#pragma once
#include "AdvancedEntitySystem.h"
#include "ConstraintComponent.h"

// =======================================================================================
//                                      ConstraintSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # ConstraintSystem
/// 
/// 15-7-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ConstraintSystem : public AdvancedEntitySystem
{
private:
	artemis::ComponentMapper<ConstraintComponent> constraintMapper;
public:

	ConstraintSystem(btDiscreteDynamicsWorld* p_dynamicsWorld)
	{
		m_dynamicsWorldPtr = p_dynamicsWorld;
		addComponentType<ConstraintComponent>();
	};

	virtual void initialize()
	{
		constraintMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e)
	{
		ConstraintComponent* constraint = constraintMapper.get(e);
		if (constraint)
			constraint->forceRemove(m_dynamicsWorldPtr);
	}

	virtual void added(artemis::Entity &e)
	{

	}

	virtual void processEntity(artemis::Entity &e)
	{

	}

protected:
private:
	btDiscreteDynamicsWorld* m_dynamicsWorldPtr;
};