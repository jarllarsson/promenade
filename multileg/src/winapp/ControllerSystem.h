#pragma once
#include <Artemis.h>
#include "TransformComponent.h"
#include "RigidBodyComponent.h"
#include <btBulletDynamicsCommon.h>
#include "ConstraintComponent.h"
#include <vector>
#include "ControllerComponent.h"
#include <ToString.h>
#include <DebugPrint.h>
#include <MathHelp.h>
#include <ppl.h>
// =======================================================================================
//                                 ControllerSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	The specialized controller system that builds the controllers and
///			inits the kernels and gathers their results
///        
/// # ControllerSystem
/// 
/// 20-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

//#define MULTI

class ControllerSystem : public artemis::EntityProcessingSystem
{
private:
	artemis::ComponentMapper<ControllerComponent> controllerComponentMapper;
	std::vector<ControllerComponent*> m_controllersToBuild;
	std::vector<ControllerComponent*> m_controllers;
	std::vector<glm::vec3> m_torques;
	std::vector<btRigidBody*> m_rigidBodies;
	std::vector<glm::mat4> m_transforms;
public:
	ControllerSystem()
	{
		addComponentType<ControllerComponent>();
		//addComponentType<RigidBodyComponent>();
		//m_dynamicsWorldPtr = p_dynamicsWorld;
	};

	virtual void initialize()
	{
		controllerComponentMapper.init(*world);
		//rigidBodyMapper.init(*world);
	};

	virtual void removed(artemis::Entity &e);

	virtual void added(artemis::Entity &e);

	virtual void processEntity(artemis::Entity &e);

	void start(float p_dt);

	void finish();

	void applyTorques();

	// Build uninited controllers, this has to be called 
	// after constraints & rb's have been inited by their systems
	void buildCheck();
private:
	unsigned int addJoint(RigidBodyComponent* p_joint);
	void setTransformMatrixFromRigidBodyInList(unsigned int p_idx);
	glm::vec3 DOFAxisByVecCompId(unsigned int p_id);
};

void ControllerSystem::removed(artemis::Entity &e)
{

}

void ControllerSystem::added(artemis::Entity &e)
{
	ControllerComponent* controller = controllerComponentMapper.get(e);

	m_controllersToBuild.push_back(controller);
}

void ControllerSystem::processEntity(artemis::Entity &e)
{

}

void ControllerSystem::start(float p_dt)
{
	//DEBUGPRINT(( (std::string("\nController start DT=") + toString(p_dt) + "\n").c_str() ));

	// Update all transforms
	for (int i = 0; i < m_rigidBodies.size(); i++)
	{
		setTransformMatrixFromRigidBodyInList(i);
		m_torques[i] = glm::vec3(0.0f);
	}

	if (m_controllers.size()>0)
	{
		for (int n = 0; n < m_controllers.size();n++)
		{
			ControllerComponent* controller = m_controllers[n];
			ControllerComponent::Chain* legChain = &controller->m_DOFChain;
			// Run controller code here
#ifndef MULTI
			for (int i = 0; i < legChain->getSize(); i++)
			{
				unsigned int tIdx = legChain->jointIDXChain[i];
				glm::vec3 torqueBase = legChain->DOFChain[i];
				glm::quat rot = glm::quat(torqueBase)*glm::quat(m_transforms[tIdx]);
				m_torques[tIdx] += torqueBase*13.0f/**(float)(TORAD)*/;
			}
#else
			//concurrency::combinable<glm::vec3> sumtorques;
			concurrency::parallel_for(0, (int)legChain->getSize(), [&](int i) {
				unsigned int tIdx = legChain->jointIDXChain[i];
				glm::vec3 torqueBase = legChain->DOFChain[i];
				glm::quat rot = glm::quat(torqueBase)*glm::quat(m_transforms[tIdx]);
				m_torques[tIdx] += torqueBase*13.0f/**(float)(TORAD)*/;
			});

#endif
		}
	}
}

void ControllerSystem::finish()
{

}

void ControllerSystem::applyTorques()
{
	if (m_rigidBodies.size()==m_torques.size())
	{
		for (int i = 0; i < m_rigidBodies.size(); i++)
		{
			glm::vec3* t = &m_torques[i];
			m_rigidBodies[i]->applyTorque(btVector3(t->x, t->y, t->z));
		}
	}
}

void ControllerSystem::buildCheck()
{
	for (int i = 0; i < m_controllersToBuild.size(); i++)
	{
		ControllerComponent* controller=m_controllersToBuild[i];
		ControllerComponent::LegFrame* legFrame = &controller->m_legFrames[0];
		ControllerComponent::Chain* legChain = &controller->m_DOFChain;
		// Build the controller (Temporary code)
		// The below should be done for each leg (even the root)
		// Create ROOT
		RigidBodyComponent* root = (RigidBodyComponent*)legFrame->m_legFrameEntity->getComponent<RigidBodyComponent>();
		unsigned int rootIdx = addJoint(root); 
		glm::vec3 DOF;
		for (int n = 0; n < 3; n++)
		{
			legChain->jointIDXChain.push_back(rootIdx);
			legChain->DOFChain.push_back(DOFAxisByVecCompId(n)); // root has 3DOF (for now, to not over-optimize, we add three vec3's)
		}
		// rest of leg
		artemis::Entity* jointEntity = legFrame->m_upperLegEntity;
		while (jointEntity != NULL)
		{
			// Get joint data
			RigidBodyComponent* joint = (RigidBodyComponent*)jointEntity->getComponent<RigidBodyComponent>();
			ConstraintComponent* parentLink = (ConstraintComponent*)jointEntity->getComponent<ConstraintComponent>();
			// Add the joint
			unsigned int idx = addJoint(joint);
			// Get DOF on joint
			const glm::vec3* lims = parentLink->getDesc()->m_angularDOF_LULimits;
			for (int n = 0; n < 3; n++) // go through all DOFs and add if free
			{
				// check if upper limit is greater than lower limit, componentwise.
				// If true, add as DOF
				if (lims[0][n] < lims[1][n])
				{
					legChain->jointIDXChain.push_back(idx);
					legChain->DOFChain.push_back(DOFAxisByVecCompId(n));
				}
			}
			// Get child joint for next iteration
			ConstraintComponent* childLink = joint->getChildConstraint(0);
			if (childLink != NULL)
				jointEntity = childLink->getOwnerEntity();
			else
				jointEntity = NULL;
		}
		//
		m_controllers.push_back(controller);
	}
	m_controllersToBuild.clear();
}

unsigned int ControllerSystem::addJoint(RigidBodyComponent* p_joint)
{
	m_rigidBodies.push_back(p_joint->getRigidBody());
	m_torques.resize(m_rigidBodies.size());
	m_transforms.resize(m_rigidBodies.size());
	unsigned int idx = m_rigidBodies.size() - 1;
	setTransformMatrixFromRigidBodyInList(idx);
	return idx; // return idx of inserted
}

glm::vec3 ControllerSystem::DOFAxisByVecCompId(unsigned int p_id)
{
	if (p_id == 0)
		return glm::vec3(1.0f, 0.0f, 0.0f);
	else if (p_id == 1)
		return glm::vec3(0.0f, 1.0f, 0.0f);
	else
		return glm::vec3(0.0f, 0.0f, 1.0f);
}

void ControllerSystem::setTransformMatrixFromRigidBodyInList(unsigned int p_idx)
{
	if (p_idx<m_rigidBodies.size() && m_transforms.size()==m_rigidBodies.size())
	{
		btRigidBody* body = m_rigidBodies[p_idx];
		if (body != NULL/* && body->isInWorld() && body->isActive()*/)
		{
			btMotionState* motionState = body->getMotionState();
			btTransform physTransform;
			motionState->getWorldTransform(physTransform);
			// Get the transform from Bullet and into mat
			glm::mat4 mat;
			physTransform.getOpenGLMatrix(glm::value_ptr(mat));
			m_transforms[p_idx] = mat;
		}
	}
}
