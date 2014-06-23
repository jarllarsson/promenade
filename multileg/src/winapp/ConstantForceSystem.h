#pragma once
#include "AdvancedEntitySystem.h"
#include "ConstantForceComponent.h"
#include "RigidBodyComponent.h"

// =======================================================================================
//                                      ConstantForceSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # ConstantForceSystem
/// 
/// 20-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ConstantForceSystem : public AdvancedEntitySystem
{
private:
	artemis::ComponentMapper<ConstantForceComponent> cforceMapper;
	artemis::ComponentMapper<RigidBodyComponent> rigidBodyMapper;
	std::vector<artemis::Entity*> m_entities;
public:

	ConstantForceSystem()
	{
		addComponentType<ConstantForceComponent>();
		addComponentType<RigidBodyComponent>();
	};

	virtual void initialize()
	{
		cforceMapper.init(*world);
		rigidBodyMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e)
	{

	}

	virtual void added(artemis::Entity &e)
	{
		m_entities.push_back(&e);
		// NOT PERFECT, STILL NEED TO HANDLE WHEN
		// COMPONENTS AND ENTITIES ARE REMOVED
	}

	virtual void processEntity(artemis::Entity &e)
	{
		
	}

	virtual void fixedUpdate(float p_dt)
	{
		for (int i = 0; i < m_entities.size(); i++)
		{
			artemis::Entity& e = *m_entities[i];
			RigidBodyComponent* rigidBody = rigidBodyMapper.get(e);
			ConstantForceComponent* cforce = cforceMapper.get(e);
			if (rigidBody && cforce && rigidBody->isInited())
			{
				btRigidBody* body = rigidBody->getRigidBody();
				if (body != NULL)
				{								
					glm::vec3& v = cforce->m_force;				
					rigidBody->getRigidBody()->applyCentralForce(btVector3(v.x, v.y, v.z));
					if (cforce->m_timeBound)
					{
						cforce->m_time -= p_dt;
						if (cforce->m_time <= 0.0f)
							e.removeComponent<ConstantForceComponent>();
					}
				}
			}
		}
		
	}

protected:
private:
};