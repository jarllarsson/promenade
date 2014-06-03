#include "ControllerSystem.h"

#include <ppl.h>
#include <ToString.h>
#include <DebugPrint.h>
#include <MathHelp.h>
#include <btBulletDynamicsCommon.h>
#include "ConstraintComponent.h"


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

void ControllerSystem::update(float p_dt)
{
	//DEBUGPRINT(( (std::string("\nController start DT=") + toString(p_dt) + "\n").c_str() ));
	m_runTime += p_dt;

	// Update all transforms
	for (int i = 0; i < m_jointRigidBodies.size(); i++)
	{
		saveJointMatrix(i);
		m_jointTorques[i] = glm::vec3(0.0f);
	}

	if (m_controllers.size()>0)
	{
		// Start with making the controllers parallel only.
		// They still write to a global torque list, but without collisions.
#ifndef MULTI
		// Single threaded implementation
		for (int n = 0; n < m_controllers.size(); n++)
		{
			ControllerComponent* controller = m_controllers[n];
			ControllerComponent::Chain* legChain = &controller->m_DOFChain;
			// Run controller code here

			for (unsigned int i = 0; i < legChain->getSize(); i++)
			{
				unsigned int tIdx = legChain->jointIDXChain[i];
				glm::vec3 torqueBase = legChain->DOFChain[i];
				glm::quat rot = glm::quat(torqueBase)*glm::quat(m_jointWorldTransforms[tIdx]);
				m_jointTorques[tIdx] += torqueBase*13.0f/**(float)(TORAD)*/;
			}
		}
#else
		// Multi threaded CPU implementation
		//concurrency::combinable<glm::vec3> sumtorques;
		concurrency::parallel_for(0, (int)legChain->getSize(), [&](int i) {
			unsigned int tIdx = legChain->jointIDXChain[i];
			glm::vec3 torqueBase = legChain->DOFChain[i];
			glm::quat rot = glm::quat(torqueBase)*glm::quat(m_jointWorldTransforms[tIdx]);
			m_jointTorques[tIdx] += torqueBase*13.0f/**(float)(TORAD)*/;
		});

#endif

	}
}

void ControllerSystem::finish()
{

}

void ControllerSystem::applyTorques()
{
	if (m_jointRigidBodies.size() == m_jointTorques.size())
	{
		for (int i = 0; i < m_jointRigidBodies.size(); i++)
		{
			glm::vec3* t = &m_jointTorques[i];
			m_jointRigidBodies[i]->applyTorque(btVector3(t->x, t->y, t->z));
		}
	}
}

void ControllerSystem::buildCheck()
{
	for (int i = 0; i < m_controllersToBuild.size(); i++)
	{
		ControllerComponent* controller = m_controllersToBuild[i];
		ControllerComponent::LegFrameEntityConstruct* legFrameEntities = controller->getLegFrameEntityConstruct(0);
		ControllerComponent::Chain* legChain = &controller->m_DOFChain;
		// Build the controller (Temporary code)
		// The below should be done for each leg (even the root)
		// Create ROOT
		RigidBodyComponent* rootRB = (RigidBodyComponent*)legFrameEntities->m_legFrameEntity->getComponent<RigidBodyComponent>();
		TransformComponent* rootTransform = (TransformComponent*)legFrameEntities->m_legFrameEntity->getComponent<TransformComponent>();
		unsigned int rootIdx = addJoint(rootRB, rootTransform);
		glm::vec3 DOF;
		for (int n = 0; n < 3; n++)
		{
			legChain->jointIDXChain.push_back(rootIdx);
			legChain->DOFChain.push_back(DOFAxisByVecCompId(n)); // root has 3DOF (for now, to not over-optimize, we add three vec3's)
		}
		// rest of leg
		artemis::Entity* jointEntity = legFrameEntities->m_upperLegEntities[0];
		while (jointEntity != NULL)
		{
			// Get joint data
			TransformComponent* jointTransform = (TransformComponent*)jointEntity->getComponent<TransformComponent>();
			RigidBodyComponent* jointRB = (RigidBodyComponent*)jointEntity->getComponent<RigidBodyComponent>();
			ConstraintComponent* parentLink = (ConstraintComponent*)jointEntity->getComponent<ConstraintComponent>();
			// Add the joint
			unsigned int idx = addJoint(jointRB, jointTransform);
			// Get DOF on joint
			const glm::vec3* lims = parentLink->getDesc()->m_angularDOF_LULimits;
			for (int n = 0; n < 3; n++) // go through all DOFs and add if free
			{
				// check if upper limit is greater than lower limit, component-wise.
				// If true, add as DOF
				if (lims[0][n] < lims[1][n])
				{
					legChain->jointIDXChain.push_back(idx);
					legChain->DOFChain.push_back(DOFAxisByVecCompId(n));
				}
			}
			// Get child joint for next iteration
			ConstraintComponent* childLink = jointRB->getChildConstraint(0);
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

unsigned int ControllerSystem::addJoint(RigidBodyComponent* p_jointRigidBody, TransformComponent* p_jointTransform)
{
	m_jointRigidBodies.push_back(p_jointRigidBody->getRigidBody());
	m_jointTorques.resize(m_jointRigidBodies.size());
	glm::mat4 matPosRot = p_jointTransform->getMatrixPosRot();
	m_jointWorldTransforms.push_back(matPosRot);
	m_jointLengths.push_back(p_jointTransform->getScale().y);
	// m_jointWorldTransforms.resize(m_jointRigidBodies.size());
	// m_jointLengths.resize(m_jointRigidBodies.size());
	m_jointWorldEndpoints.resize(m_jointRigidBodies.size());
	unsigned int idx = (unsigned int)(m_jointRigidBodies.size() - 1);
	saveJointWorldEndpoint(idx, matPosRot);
	// saveJointMatrix(idx);
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

void ControllerSystem::saveJointMatrix(unsigned int p_rigidBodyIdx)
{
	unsigned int idx = p_rigidBodyIdx;
	if (idx < m_jointRigidBodies.size() && m_jointWorldTransforms.size() == m_jointRigidBodies.size())
	{
		btRigidBody* body = m_jointRigidBodies[idx];
		if (body != NULL/* && body->isInWorld() && body->isActive()*/)
		{
			btMotionState* motionState = body->getMotionState();
			btTransform physTransform;
			motionState->getWorldTransform(physTransform);
			// Get the transform from Bullet and into mat
			glm::mat4 mat;
			physTransform.getOpenGLMatrix(glm::value_ptr(mat));
			m_jointWorldTransforms[idx] = mat; // note, use same index for transform list
			saveJointWorldEndpoint(idx, mat);
		}
	}
}


void ControllerSystem::saveJointWorldEndpoint(unsigned int p_idx, glm::mat4& p_worldMatPosRot)
{
	m_jointWorldEndpoints[p_idx] = glm::vec4(0.0f, m_jointLengths[p_idx], 0.0f, 1.0f)*p_worldMatPosRot;
}



void ControllerSystem::controllerUpdate(int p_controllerId, float p_dt)
{
	float dt = p_dt;
	// m_currentVelocity = transform.position - m_oldPos;
	//calcHeadAcceleration();

	// Advance the player
	//m_player.updatePhase(dt);

	// Update desired velocity
	//updateDesiredVelocity(dt);

	// update feet positions
	//updateFeet();

	// Recalculate all torques for this frame
	//updateTorques(dt);

	// Debug color of legs when in stance
	//debugColorLegs();

	//m_oldPos = transform.position;
}

