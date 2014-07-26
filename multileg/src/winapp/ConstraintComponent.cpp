#include "ConstraintComponent.h"


const ConstraintComponent::ConstraintDesc* ConstraintComponent::getDesc() const
{
	return m_desc;
}

void ConstraintComponent::init(btGeneric6DofConstraint* p_constraint, artemis::Entity* p_ownerEntity)
{
	m_constraint = p_constraint;
	m_owner = p_ownerEntity;
	m_inited = true;
}

bool ConstraintComponent::isInited()
{
	return m_inited;
}

artemis::Entity* ConstraintComponent::getParent()
{
	return m_parent;
}

btGeneric6DofConstraint* ConstraintComponent::getConstraint()
{
	return m_constraint;
}

void ConstraintComponent::forceRemove(btDiscreteDynamicsWorld* p_world)
{
	if (!m_removed)
	{
		if (p_world!=NULL && m_constraint!=NULL)
			p_world->removeConstraint(m_constraint);
		SAFE_DELETE(m_constraint);
		m_removed = true;
	}
}

artemis::Entity* ConstraintComponent::getOwnerEntity()
{
	return m_owner;
}
