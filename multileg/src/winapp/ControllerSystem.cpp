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
	int controllerCount = m_controllers.size();
	if (m_controllers.size()>0)
	{
		// Start with making the controllers parallel only.
		// They still write to a global torque list, but without collisions.
#ifndef MULTI
		// Single threaded implementation
		for (int n = 0; n < controllerCount; n++)
		{
			ControllerComponent* controller = m_controllers[n];
			ControllerComponent::VFChain* legChain = &controller->m_DOFChain;
			// Run controller code here
			controllerUpdate(n, p_dt);
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
		concurrency::parallel_for(0, controllerCount, [&](int n) {
			ControllerComponent* controller = m_controllers[n];
			ControllerComponent::VFChain* legChain = &controller->m_DOFChain;
			// Run controller code here
			controllerUpdate(n, p_dt);
			for (unsigned int i = 0; i < legChain->getSize(); i++)
			{
				unsigned int tIdx = legChain->jointIDXChain[i];
				glm::vec3 torqueBase = legChain->DOFChain[i];
				glm::quat rot = glm::quat(torqueBase)*glm::quat(m_jointWorldTransforms[tIdx]);
				//m_jointTorques[tIdx] += torqueBase*13.0f/**(float)(TORAD)*/;
			}
		});
		/*concurrency::parallel_for(0, (int)legChain->getSize(), [&](int i) {
			unsigned int tIdx = legChain->jointIDXChain[i];
			glm::vec3 torqueBase = legChain->DOFChain[i];
			glm::quat rot = glm::quat(torqueBase)*glm::quat(m_jointWorldTransforms[tIdx]);
			m_jointTorques[tIdx] += torqueBase*13.0f;
		});
	*/

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
		ControllerComponent::LegFrame* legFrame = controller->getLegFrame(0);
		// start by storing the current torque list size as offset, this'll be where we'll begin this
		// controller's chunk of the torque list
		unsigned int torqueListOffset = m_jointTorques.size();
		// Build the controller (Temporary code)
		// The below should be done for each leg (even the root)
		// Create ROOT
		RigidBodyComponent* rootRB = (RigidBodyComponent*)legFrameEntities->m_legFrameEntity->getComponent<RigidBodyComponent>();
		TransformComponent* rootTransform = (TransformComponent*)legFrameEntities->m_legFrameEntity->getComponent<TransformComponent>();
		unsigned int rootIdx = addJoint(rootRB, rootTransform);
		legFrame->m_legFrameJointId = rootIdx; // store idx to root for leg frame
		// prepare legs			
		legFrame->m_legs.resize(legFrameEntities->m_upperLegEntities.size()); // Allocate the number of specified legs
		unsigned int legCount = legFrame->m_legs.size();
		for (int x = 0; x < legCount; x++)
		{
			// start by adding the already existing root id (needed in all leg chains)
			addJointToChain(&legFrame->m_legs[x], rootIdx);
			// Traverse the segment structure for the leg to get the rest
			artemis::Entity* jointEntity = legFrameEntities->m_upperLegEntities[x];
			while (jointEntity != NULL)
			{
				// Get joint data
				TransformComponent* jointTransform = (TransformComponent*)jointEntity->getComponent<TransformComponent>();
				RigidBodyComponent* jointRB = (RigidBodyComponent*)jointEntity->getComponent<RigidBodyComponent>();
				ConstraintComponent* parentLink = (ConstraintComponent*)jointEntity->getComponent<ConstraintComponent>();
				// Add the joint
				unsigned int idx = addJoint(jointRB, jointTransform);
				// Get DOF on joint to chain
				addJointToChain(&legFrame->m_legs[x], idx, parentLink->getDesc()->m_angularDOF_LULimits);
				// Get child joint for next iteration
				ConstraintComponent* childLink = jointRB->getChildConstraint(0);
				if (childLink != NULL)
					jointEntity = childLink->getOwnerEntity();
				else
					jointEntity = NULL;
			}
		}
		// Calculate number of torques axes in list, store
		unsigned int torqueListChunkSize = m_jointTorques.size() - torqueListOffset;
		controller->setTorqueListProperties(torqueListOffset, torqueListChunkSize);
		// Add
		m_controllers.push_back(controller);
		initControllerVelocityStat(m_controllers.size() - 1);
	}
	m_controllersToBuild.clear();
}

void ControllerSystem::addJointToChain(ControllerComponent::Leg* p_leg, unsigned int p_idx, const glm::vec3* p_angularLims)
{
	ControllerComponent::VFChain* legChain = &p_leg->m_DOFChain;
	// root has 3DOF (for now, to not over-optimize, we add three vec3's)
	for (int n = 0; n < 3; n++)
	{
		if (p_angularLims == NULL || (p_angularLims[0][n] < p_angularLims[1][n]))
		{
			legChain->jointIDXChain.push_back(p_idx);
			legChain->DOFChain.push_back(DOFAxisByVecCompId(n));
		}
	}
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
	ControllerComponent* controller = m_controllers[p_controllerId];
	// m_currentVelocity = transform.position - m_oldPos;
	//calcHeadAcceleration();

	// Advance the player
	//m_player.updatePhase(dt);
	controller->m_player.updatePhase(dt);

	// Update desired velocity
	updateVelocityStats(p_controllerId, controller, p_dt);

	// update feet positions
	updateFeet(p_controllerId, controller);

	// Recalculate all torques for this frame
	updateTorques(p_controllerId, controller, dt);

	// Debug color of legs when in stance
	//debugColorLegs();

	//m_oldPos = transform.position;
}
void ControllerSystem::updateVelocityStats(int p_controllerId, ControllerComponent* p_controller, float p_dt)
{
	glm::vec3 pos = getControllerPosition(p_controller);
	// Update the current velocity
	glm::vec3 currentV = pos - m_controllerVelocityStats[p_controllerId].m_oldPos;
	m_controllerVelocityStats[p_controllerId].m_currentVelocity = currentV;
	// Store this position
	m_controllerVelocityStats[p_controllerId].m_oldPos = pos;
	// Calculate the desired velocity needed in order to reach the goal
	// velocity from the current velocity
	// Function for deciding the current desired velocity in order
	// to reach the goal velocity
	glm::vec3 goalV = m_controllerVelocityStats[p_controllerId].m_goalVelocity;
	glm::vec3 desiredV = m_controllerVelocityStats[p_controllerId].m_desiredVelocity;
	float goalSqrMag = glm::sqrLength(goalV);
	float currentSqrMag = glm::sqrLength(currentV);
	float stepSz = 0.5f * p_dt;
	// Note the material doesn't mention taking dt into 
	// account for the step size, they might be running fixed timestep
	// Here the dt received is the time since we last ran the control logic
	//
	// If the goal is faster
	if (goalSqrMag > currentSqrMag)
	{
		// Take steps no bigger than 0.5m/s
		if (goalSqrMag >= currentSqrMag + stepSz)
			desiredV = goalV;
		else
			desiredV += glm::normalize(currentV) * stepSz;
	}
	else // if the goal is slower
	{
		// Take steps no smaller than 0.5
		if (goalSqrMag <= currentSqrMag - stepSz)
			desiredV = goalV;
		else
			desiredV -= glm::normalize(currentV) * stepSz;
	}
	m_controllerVelocityStats[p_controllerId].m_desiredVelocity = desiredV;
}


void ControllerSystem::initControllerVelocityStat(unsigned int p_idx)
{
	glm::vec3 pos = getControllerPosition(p_idx);
	VelocityStat vstat{ pos, glm::vec3(0.0f), glm::vec3(0.0f) };
	m_controllerVelocityStats.push_back(vstat);
}

glm::vec3 ControllerSystem::getControllerPosition(unsigned int p_controllerId)
{
	ControllerComponent* controller = m_controllers[p_controllerId];
	return getControllerPosition(controller);
}

glm::vec3 ControllerSystem::getControllerPosition(ControllerComponent* p_controller)
{
	unsigned int legFrameJointId = p_controller->getLegFrame(0)->m_legFrameJointId;
	glm::vec3 pos = MathHelp::toVec3(m_jointWorldTransforms[legFrameJointId] * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
	return pos;
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

void ControllerSystem::updateFeet( int p_controllerId, ControllerComponent* p_controller )
{
	//for (int i = 0; i < m_legFrames.Length; i++)
	//{
	//	m_legFrames[i].updateReferenceFeetPositions(m_player.m_gaitPhase, m_time, m_goalVelocity);
	//	m_legFrames[i].updateFeet(m_player.m_gaitPhase, m_currentVelocity, m_desiredVelocity);
	//	//m_legFrames[i].tempApplyFootTorque(m_player.m_gaitPhase);
	//}
}

void ControllerSystem::updateTorques(int p_controllerId, ControllerComponent* p_controller, float p_dt)
{
	float phi = p_controller->m_player.getPhase();
	unsigned int torqueCount = p_controller->getTorqueListChunkSize();
	unsigned int torqueIdxOffset = p_controller->getTorqueListOffset();
	//// Get the three variants of torque
	//Vector3[] tPD = computePDTorques(phi);
	//Vector3[] tCGVF = computeCGVFTorques(phi, p_dt);
	std::vector<glm::vec3> tPD(torqueCount);
	std::vector<glm::vec3> tCGVF(torqueCount);
	std::vector<glm::vec3> tVF(torqueCount);
	//
	//computePDTorques(&tPD, phi);
	computeVFTorques(&tVF, p_controller, phi, p_dt);
	//computeCGVFTorques(&tCGVF, phi, p_dt);
	////// Sum them
	for (int i = 0; i < torqueCount; i++)
	{
		m_jointTorques[torqueIdxOffset+i] = /*tPD[i] + */tVF[i] /*+ tCGVF[i]*/;
	}
	//
	// Apply them to the leg frames, also
	// feed back corrections for hip joints
	for (int i = 0; i < p_controller->getLegFrameCount(); i++)
	{
		//applyNetLegFrameTorque(p_controllerId, p_controller, i, &m_jointTorques, torqueIdxOffset, torqueCount, phi);
	}
}

void ControllerSystem::computeVFTorques(std::vector<glm::vec3>* p_outTVF, ControllerComponent* p_controller, float p_phi, float p_dt)
{
	if (m_useVFTorque)
	{
		for (int i = 0; i < p_controller->getLegFrameCount(); i++)
		{
			ControllerComponent::LegFrame* lf = p_controller->getLegFrame(i);
			lf.calculateNetLegVF(p_phi, p_dt, m_currentVelocity, m_desiredVelocity);
			// Calculate torques using each leg chain
			for (int n = 0; n < LegFrame.c_legCount; n++)
			{
				//  get the joints
				int legFrameRoot = lf.m_id;
				//legFrameRoot = -1;
				int legRoot = lf.m_neighbourJointIds[n];
				int legSegmentCount = LegFrame.c_legSegments; // hardcoded now
				// Use joint ids to get dof ids
				// Start in chain
				int legFrameRootDofId = -1; // if we have separate root as base link
				if (legFrameRoot != -1) legFrameRootDofId = m_chain[legFrameRoot].m_dofListIdx;
				// otherwise, use first in chain as base link
				int legRootDofId = m_chain[legRoot].m_dofListIdx;
				// end in chain
				int lastDofIdx = legRoot + legSegmentCount - 1;
				int legDofEnd = m_chain[lastDofIdx].m_dofListIdx + m_chain[lastDofIdx].m_dof.Length;
				//
				// get force for the leg
				Vector3 VF = lf.m_netLegBaseVirtualForces[n];
				// Calculate torques for each joint
				// Start by updating joint information based on their gameobjects
				Vector3 end = transform.localPosition;
				//Debug.Log("legroot "+legRoot+" legseg "+legSegmentCount);
				int jointstart = legRoot;
				if (legFrameRoot != -1) jointstart = legFrameRoot;
				for (int x = jointstart; x < legRoot + legSegmentCount; x++)
				{
					if (legFrameRoot != -1 && x < legRoot && x != legFrameRoot)
						x = legRoot;
					Joint current = m_chain[x];
					GameObject currentObj = m_chainObjs[x];
					//Debug.Log("joint pos: " + currentObj.transform.localPosition);
					// Update Joint
					current.length = currentObj.transform.localScale.y;
					current.m_position = currentObj.transform.position /*- (-currentObj.transform.up) * current.length * 0.5f*/;
					current.m_endPoint = currentObj.transform.position + (-currentObj.transform.up) * current.length/* * 0.5f*/;
					//m_chain[i] = current;
					//Debug.DrawLine(current.m_position, current.m_endPoint, Color.red);
					//Debug.Log(x+" joint pos: " + current.m_position + " = " + m_chain[x].m_position);
					end = current.m_endPoint;
				}
				//foreach(Joint j in m_chain)
				//    Debug.Log("joint pos CC: " + j.m_position);

				//CMatrix J = Jacobian.calculateJacobian(m_chain, m_chain.Count, end, Vector3.forward);
				CMatrix J = Jacobian.calculateJacobian(m_chain,     // Joints (Joint script)
					m_chainObjs, // Gameobjects in chain
					m_dofs,      // Degrees Of Freedom (Per joint)
					m_dofJointId,// Joint id per DOF 
					end + VF,    // Target position
					legRootDofId,// Starting link id in chain (start offset)
					legDofEnd,  // End of chain of link (ie. size)
					legFrameRootDofId); // As we use the leg frame as base, we supply it separately (it will be actual root now)
				CMatrix Jt = CMatrix.Transpose(J);

				//Debug.DrawLine(end, end + VF, Color.magenta, 0.3f);
				int jIdx = 0;
				int extra = 0;
				int start = legRootDofId;
				if (legFrameRootDofId >= 0)
				{
					start = legFrameRootDofId;
					extra = m_chain[legFrameRoot].m_dof.Length;
				}


				for (int g = start; g < legDofEnd; g++)
				{
					if (extra > 0)
						extra--;
					else if (g < legRootDofId)
						g = legRootDofId;

					// store torque
					int x = m_dofJointId[g];
					Vector3 addT = m_dofs[g] * Vector3.Dot(new Vector3(Jt[jIdx, 0], Jt[jIdx, 1], Jt[jIdx, 2]), VF);
					newTorques[x] += addT;
					jIdx++;
					//Vector3 drawTorque = new Vector3(0.0f, 0.0f, -addT.x);
					//Debug.DrawLine(m_joints[x].transform.position, m_joints[x].transform.position + drawTorque*0.1f, Color.cyan);

				}
				// Come to think of it, the jacobian and torque could be calculated in the same
				// kernel as it lessens write to global memory and the need to fetch joint matrices several time (transform above)
			}
		}
	}
}

