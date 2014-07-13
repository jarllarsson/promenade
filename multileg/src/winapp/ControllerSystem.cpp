#include "ControllerSystem.h"

#include <ppl.h>
#include <ToString.h>
#include <DebugPrint.h>
#include <MathHelp.h>
#include <btBulletDynamicsCommon.h>
#include "ConstraintComponent.h"
#include "JacobianHelper.h"
#include "MaterialComponent.h"
#include "Time.h"
#include "PhysWorldDefines.h"
#include "RenderComponent.h"
#include "PositionRefComponent.h"


ControllerSystem::~ControllerSystem()
{

}



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
	// Regular processing here
	// Perfect for debugging
	// Non-optional
	ControllerComponent* controller = controllerComponentMapper.get(e);
	if (controller != NULL && controller->isBuildComplete())
	{
		ControllerComponent::LegFrame* lf = controller->getLegFrame(0);
		unsigned int legCount = (unsigned int)lf->m_legs.size();
		for (unsigned int i = 0; i < legCount; i++)
		{
			unsigned int jointId = lf->m_hipJointId[i];
			if (isInControlledStance(lf, i, controller->m_player.getPhase()))
			{
				for (int i = 0; i < 3; i++) // 3 segments
				{
					artemis::Entity* segEntity = m_dbgJointEntities[jointId + i];
					MaterialComponent* mat = (MaterialComponent*)segEntity->getComponent<MaterialComponent>();
					if (mat)
					{
						mat->highLight();
					}
				}
			}
		}
	}
}

void ControllerSystem::fixedUpdate(float p_dt)
{
	DEBUGPRINT(( (std::string("\nDT=") + ToString(p_dt) + "\n").c_str() ));
	m_runTime += p_dt;
	m_steps++;

	double startTiming = Time::getTimeMs();

	// Clear debug draw batch 
	// (not optimal to only do it here if drawing from game systems,
	// batch calls should be put in a map or equivalent)
	dbgDrawer()->clearDrawCalls();

	// Update all transforms
	for (unsigned int i = 0; i < m_jointRigidBodies.size(); i++)
	{
		saveJointMatrix(i);
		m_jointTorques[i] = glm::vec3(0.0f);
	}
	int controllerCount = (int)m_controllers.size();
	if (m_controllers.size()>0)
	{
		// First, we have to read collision status for all feet. SILLY
		for (int n = 0; n < controllerCount; n++)
		{
			ControllerComponent* controller = m_controllers[(unsigned int)n];
			writeFeetCollisionStatus(controller);
		}

		// Start with making the controllers parallel only.
		// They still write to a global torque list, but without collisions.
#ifndef MULTI
		// Single threaded implementation
		for (int n = 0; n < controllerCount; n++)
		{
			ControllerComponent* controller = m_controllers[(unsigned int)n];
			// Run controller code here
			controllerUpdate((unsigned int)n, p_dt);
		}
#else
		// Multi threaded CPU implementation
		//concurrency::combinable<glm::vec3> sumtorques;
		concurrency::parallel_for(0, controllerCount, [&](int n) {
			ControllerComponent* controller = m_controllers[n];
			// Run controller code here
			controllerUpdate(n, p_dt);
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
	else
	{
		DEBUGPRINT(("\nNO CONTROLLERS YET\n"));
	}

	if (m_perfRecorder != NULL)
		m_perfRecorder->saveMeasurement(Time::getTimeMs() - startTiming,m_steps);
}

void ControllerSystem::finish()
{

}

void ControllerSystem::applyTorques( float p_dt )
{
	if (m_jointRigidBodies.size() == m_jointTorques.size())
	{
		float tLim = 100.0f;
		for (unsigned int i = 0; i < m_jointRigidBodies.size(); i++)
		{
			glm::vec3* t = &m_jointTorques[i];
			if (glm::length(*t)>tLim) 
				*t = glm::normalize(*t)*tLim;
			m_jointRigidBodies[i]->applyTorque(btVector3(t->x, t->y, t->z));
		}
	}
}

void ControllerSystem::buildCheck()
{
	for (unsigned int i = 0; i < m_controllersToBuild.size(); i++)
	{
		glm::vec3 startGaitVelocity(0.0f, 0.0f, 3.0f);
		ControllerComponent* controller = m_controllersToBuild[i];
		ControllerComponent::LegFrameEntityConstruct* legFrameEntities = controller->getLegFrameEntityConstruct(0);
		ControllerComponent::LegFrame* legFrame = controller->getLegFrame(0);
		// start by storing the current torque list size as offset, this'll be where we'll begin this
		// controller's chunk of the torque list
		unsigned int torqueListOffset = (unsigned int)m_jointTorques.size();
		// Build the controller (Temporary code)
		// The below should be done for each leg (even the root)
		// Create ROOT
		RigidBodyComponent* rootRB = (RigidBodyComponent*)legFrameEntities->m_legFrameEntity->getComponent<RigidBodyComponent>();
		TransformComponent* rootTransform = (TransformComponent*)legFrameEntities->m_legFrameEntity->getComponent<TransformComponent>();
		unsigned int rootIdx = addJoint(rootRB, rootTransform);
		m_dbgJointEntities.push_back(legFrameEntities->m_legFrameEntity); // for easy debugging options
		//
		legFrame->m_legFrameJointId = rootIdx; // store idx to root for leg frame
		// prepare legs			
		unsigned int legCount = legFrameEntities->m_upperLegEntities.size();
		legFrame->m_legs.resize(legCount); // Allocate the number of specified legs
		// when and if we have a spine, add it here and store its id to leg frame
		// Might have to do this after all leg frames instead, due to the link being at different ends of spine
		legFrame->m_spineJointId = -1; // -1 means it doesn't exist
		//
		glm::vec3 legFramePos = rootTransform->getPosition(), footPos;
		for (unsigned int x = 0; x < legCount; x++)
		{
			m_VFs.push_back(glm::vec3(0.0f, 0.0f, 0.0f));	
			unsigned int vfIdx = (unsigned int)((int)m_VFs.size()-1);
			ControllerComponent::VFChain* standardDOFChain = legFrame->m_legs[x].getVFChain(ControllerComponent::VFChainType::STANDARD_CHAIN);
			// start by adding the already existing root id (needed in all leg chains)
			addJointToStandardVFChain(standardDOFChain, rootIdx, vfIdx);
			// Traverse the segment structure for the leg to get the rest
			artemis::Entity* jointEntity = legFrameEntities->m_upperLegEntities[x];
			unsigned int jointsAddedForLeg = 0;
			while (jointEntity != NULL) // read all joints
			{
				// Get joint data
				TransformComponent* jointTransform = (TransformComponent*)jointEntity->getComponent<TransformComponent>();
				RigidBodyComponent* jointRB = (RigidBodyComponent*)jointEntity->getComponent<RigidBodyComponent>();
				ConstraintComponent* parentLink = (ConstraintComponent*)jointEntity->getComponent<ConstraintComponent>();
				// Add the joint
				unsigned int idx = addJoint(jointRB, jointTransform);
				m_dbgJointEntities.push_back(jointEntity); // for easy debugging options
				// Get DOF on joint to chain
				addJointToStandardVFChain(standardDOFChain,
										  idx, vfIdx, parentLink->getDesc()->m_angularDOF_LULimits);
				// Register joint for PD (and create PD)
				addJointToPDChain(legFrame->m_legs[x].getPDChain(), idx, legFrame->m_legPDsKp, legFrame->m_legPDsKd);
				// Get child joint for next iteration
				ConstraintComponent* childLink = jointRB->getChildConstraint(0);
				// Add hip joint if first
				if (jointsAddedForLeg == 0) legFrame->m_hipJointId.push_back(idx);
				// find out what to do next time
				if (childLink != NULL)
				{	// ===LEG SEGMENT===
					jointEntity = childLink->getOwnerEntity();
				}
				else // we are at the foot, so trigger termination add foot id to the leg frame
				{	// ===FOOT===
					legFrame->m_feetJointId.push_back(idx);
					footPos = jointTransform->getPosition(); // store foot position (only used to determine character height, so doesn't matter which one)
					legFrame->createFootPlacementModelVarsForNewLeg(footPos);
					// add rigidbody so we can check for collisions and late foot strikes
					m_footRigidBodyRefs.push_back(jointRB);
					legFrame->m_footRigidBodyIdx.push_back(m_footRigidBodyRefs.size() - 1);
					//
					jointEntity = NULL;
				}
				jointsAddedForLeg++;
			}
			// Copy all DOFs for ordinary VF-chain to the gravity compensation chain,
			// Then re-append a copy of decreasing size (of one) for each segment.
			// So that we get the structure of that chain as described in struct Leg
			legFrame->m_legs[x].m_DOFChainGravityComp = legFrame->m_legs[x].m_DOFChain;			
			unsigned int origGCDOFsz = legFrame->m_legs[x].m_DOFChainGravityComp.getSize();
			// Change the VFs for this list, as they need to be used to counter gravity
			// They're also static, so we only need to do this once
			unsigned int oldJointGCIdx = -1;
			vfIdx = 0;
			for (unsigned int m = 0; m < origGCDOFsz; m++)
			{
				unsigned int jointId = legFrame->m_legs[x].m_DOFChainGravityComp.m_jointIdxChain[m];
				if (jointId != oldJointGCIdx)
				{
					float mass = m_jointMass[jointId];
					m_VFs.push_back(-mass*glm::vec3(0.0f, WORLD_GRAVITY, 0.0f));
					vfIdx = (unsigned int)((int)m_VFs.size() - 1);
				}
				legFrame->m_legs[x].m_DOFChainGravityComp.m_vfIdxList[m] = vfIdx;
				oldJointGCIdx = jointId;
			}
			// Now, re-append piece by piece, so we "automatically" get the additive loop later
			for (unsigned int n = 0; n < jointsAddedForLeg; n++)
			{ 
				// n+1 to skip re-appending of root:
				repeatAppendChainPart(&legFrame->m_legs[x].m_DOFChainGravityComp, n + 1, jointsAddedForLeg - n, origGCDOFsz);
			}
		}
		legFrame->m_height = legFramePos.y - footPos.y;
		legFrame->m_heightLFTraj.reset(PieceWiseLinear::FULL, legFrame->m_height);
		// Calculate number of torques axes in list, store
		unsigned int torqueListChunkSize = m_jointTorques.size() - torqueListOffset;
		controller->setTorqueListProperties(torqueListOffset, torqueListChunkSize);
		// Add
		controller->setToBuildComplete();
		m_controllers.push_back(controller);
		initControllerLocationAndVelocityStat((int)m_controllers.size() - 1, startGaitVelocity);
		// Finally, when all vars and lists have been built, add debug data
		// Add debug tracking for leg frame
		dbgToolbar()->addReadOnlyVariable(Toolbar::CHARACTER, "Gait phase", Toolbar::FLOAT, (const void*)(controller->m_player.getPhasePointer()), " group='LegFrame'");
		// Velocity debug
		unsigned int vlistpos = m_controllerVelocityStats.size() - 1;
		dbgToolbar()->addReadOnlyVariable(Toolbar::CHARACTER, "Current velocity", Toolbar::DIR, (const void*)&m_controllerVelocityStats[vlistpos].m_currentVelocity, " group='LegFrame'");
		dbgToolbar()->addReadOnlyVariable(Toolbar::CHARACTER, "Desired velocity", Toolbar::DIR, (const void*)&m_controllerVelocityStats[vlistpos].m_desiredVelocity, " group='LegFrame'");
		dbgToolbar()->addReadWriteVariable(Toolbar::CHARACTER, "Goal velocity", Toolbar::DIR, (void*)&m_controllerVelocityStats[vlistpos].m_goalVelocity, " group='LegFrame'");
		 // Debug, per-leg stuff
		for (unsigned int x = 0; x < legCount; x++)
		{
			bool isLeft = x == 0;
			Color3f col = (isLeft ? Color3f(1.0f, 0.0f, 0.0f) : Color3f(0.0f, 1.0f, 0.0f));
			// Add debug tracking for leg
			std::string sideName = (std::string(isLeft? "Left" : "Right") + "Leg");
			dbgToolbar()->addReadWriteVariable(Toolbar::CHARACTER, (ToString(sideName[0]) + " Duty factor").c_str(), Toolbar::FLOAT, (void*)&legFrame->m_stepCycles[x].m_tuneDutyFactor, (" group='" + sideName + "'").c_str());
			dbgToolbar()->addReadWriteVariable(Toolbar::CHARACTER, (ToString(sideName[0]) + " Step trigger").c_str(), Toolbar::FLOAT, (void*)&legFrame->m_stepCycles[x].m_tuneStepTrigger, (" group='" + sideName + "'").c_str());
			// Foot strike placement visualization
			artemis::Entity & footPlcmtDbg = world->createEntity();
			footPlcmtDbg.addComponent(new RenderComponent());
			footPlcmtDbg.addComponent(new TransformComponent());
			footPlcmtDbg.addComponent(new PositionRefComponent(&legFrame->m_footStrikePlacement[x]));
			footPlcmtDbg.addComponent(new MaterialComponent( col ));
			footPlcmtDbg.refresh();
			// Foot lift placement visualization
			artemis::Entity & footLiftDbg = world->createEntity();
			footLiftDbg.addComponent(new RenderComponent());
			footLiftDbg.addComponent(new TransformComponent(glm::vec3(0.0f),glm::vec3(0.5f,2.0f,0.5f)));
			footLiftDbg.addComponent(new PositionRefComponent(&legFrame->m_footLiftPlacement[x]));
			footLiftDbg.addComponent(new MaterialComponent(col*0.7f));
			footLiftDbg.refresh();
			// Foot target placement visualization
			artemis::Entity & footTargetDbg = world->createEntity();
			footTargetDbg.addComponent(new RenderComponent());
			footTargetDbg.addComponent(new TransformComponent(glm::vec3(0.0f), glm::vec3(0.5f, 0.5f, 0.5f),glm::quat(glm::vec3((float)HALFPI,0.0f,0.0f))));
			footTargetDbg.addComponent(new PositionRefComponent(&legFrame->m_footTarget[x]));
			footTargetDbg.addComponent(new MaterialComponent(col*0.25f));
			footTargetDbg.refresh();
		}
	}
	m_controllersToBuild.clear();
}

void ControllerSystem::addJointToStandardVFChain(ControllerComponent::VFChain* p_legVFChain, unsigned int p_idx, unsigned int p_vfIdx, const glm::vec3* p_angularLims /*= NULL*/)
{
	ControllerComponent::VFChain* legChain = p_legVFChain;
	// root has 3DOF (for now, to not over-optimize, we add three vec3's)
	for (int n = 0; n < 3; n++)
	{
		if (p_angularLims == NULL || (p_angularLims[0][n] < p_angularLims[1][n]))
		{
			legChain->m_jointIdxChain.push_back(p_idx);
			legChain->m_DOFChain.push_back(DOFAxisByVecCompId(n));
			legChain->m_vfIdxList.push_back(p_vfIdx);
		}
	}
}


void ControllerSystem::addJointToPDChain(ControllerComponent::PDChain* p_legChain, unsigned int p_idx, float p_kp, float p_kd)
{
	ControllerComponent::PDChain* legPDChain = p_legChain;
	legPDChain->m_PDChain.push_back(PDn(p_kp,p_kd));
	legPDChain->m_jointIdxChain.push_back(p_idx);
}


void ControllerSystem::repeatAppendChainPart(ControllerComponent::VFChain* p_legChain, unsigned int p_localJointOffset, 
	unsigned int p_jointCount, unsigned int p_originalChainSize)
{
	ControllerComponent::VFChain* legChain = p_legChain;
	unsigned int DOFidx=0; // current dof being considerated for copy
	unsigned int totalJointsProcessed = 0;
	unsigned int jointAddedCounter = 0; // number of added joints
	unsigned int oldJointIdx = 0;
	do 
	{
		unsigned int jointIdx=legChain->m_jointIdxChain[DOFidx];
		bool isNewJoint = (oldJointIdx != jointIdx);		
		if (isNewJoint)
			totalJointsProcessed++;
		if (totalJointsProcessed >= p_localJointOffset) // only start when we're at offset or more
		{
			legChain->m_jointIdxChain.push_back(jointIdx);
			legChain->m_DOFChain.push_back(legChain->m_DOFChain[DOFidx]);
			legChain->m_vfIdxList.push_back(legChain->m_vfIdxList[DOFidx]);
			//
			if (isNewJoint) // only increment joint counter when we're at a new joint
				jointAddedCounter++;
		}

		DOFidx++;
		oldJointIdx = jointIdx;
	} while (jointAddedCounter <= p_jointCount && DOFidx < p_originalChainSize);
	// stops adding when we have all specified joints AND have catched all its DOFs
}

unsigned int ControllerSystem::addJoint(RigidBodyComponent* p_jointRigidBody, TransformComponent* p_jointTransform)
{
	m_jointRigidBodies.push_back(p_jointRigidBody->getRigidBody());
	m_jointTorques.push_back(glm::vec3(0.0f));
	glm::mat4 matPosRot = p_jointTransform->getMatrixPosRot();
	m_jointWorldTransforms.push_back(matPosRot);
	m_jointLengths.push_back(p_jointTransform->getScale().y);
	m_jointMass.push_back(p_jointRigidBody->getMass());
	// m_jointWorldTransforms.resize(m_jointRigidBodies.size());
	// m_jointLengths.resize(m_jointRigidBodies.size());
	m_jointWorldOuterEndpoints.resize(m_jointRigidBodies.size());
	m_jointWorldInnerEndpoints.resize(m_jointRigidBodies.size());
	unsigned int idx = (unsigned int)(m_jointRigidBodies.size() - 1);
	saveJointWorldEndpoints(idx, matPosRot);
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
			//motionState->getWorldTransform(physTransform);
			physTransform = body->getWorldTransform();
			// Get the transform from Bullet and into mat
			glm::mat4 mat(0.0f);
			physTransform.getOpenGLMatrix(glm::value_ptr<glm::mediump_float>(mat));
			m_jointWorldTransforms[idx] = mat; // note, use same index for transform list
			saveJointWorldEndpoints(idx, mat);
		}
	}
}


void ControllerSystem::saveJointWorldEndpoints(unsigned int p_idx, glm::mat4& p_worldMatPosRot)
{
	// Store information on the joint's end points.
	// The outer is the one closest to a child joint.
	m_jointWorldOuterEndpoints[p_idx] = p_worldMatPosRot*glm::vec4(0.0f, -m_jointLengths[p_idx]*0.5f, 0.0f, 1.0f);
	// The inner is the one closest to the parent joint.
	m_jointWorldInnerEndpoints[p_idx] = p_worldMatPosRot*glm::vec4(0.0f, m_jointLengths[p_idx]*0.5f, 0.0f, 1.0f);
}



void ControllerSystem::controllerUpdate(unsigned int p_controllerId, float p_dt)
{
	float dt = p_dt;
	ControllerComponent* controller = m_controllers[p_controllerId];
	// m_currentVelocity = transform.position - m_oldPos;
	//calcHeadAcceleration();

	// Advance the player
	controller->m_player.updatePhase(dt);

	// Update desired velocity
	updateLocationAndVelocityStats(p_controllerId, controller, p_dt);

	// update feet positions
	updateFeet(p_controllerId, controller);

	// Recalculate all torques for this frame
	updateTorques(p_controllerId, controller, dt);

	// Debug color of legs when in stance
	//debugColorLegs();

	//m_oldPos = transform.position;
}
void ControllerSystem::updateLocationAndVelocityStats(int p_controllerId, ControllerComponent* p_controller, float p_dt)
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
	float stepSz = 0.5f/* * p_dt*/;
	// Note the material doesn't mention taking dt into 
	// account for the step size, they might be running fixed timestep
	// Here the dt received is the time since we last ran the control logic
	//
	// If the goal is faster
	if (goalSqrMag > currentSqrMag)
	{
		// Take steps no bigger than 0.5m/s
		if (goalSqrMag < currentSqrMag + stepSz)
			desiredV = goalV;
		else if (goalV != glm::vec3(0.0f))
			desiredV = currentV + glm::normalize(goalV) * stepSz;
	}
	else // if the goal is slower
	{
		// Take steps no smaller than 0.5
		if (goalSqrMag > currentSqrMag - stepSz)
			desiredV = goalV;
		else if (goalV != glm::vec3(0.0f))
			desiredV = currentV - glm::normalize(goalV) * stepSz;
	}
	m_controllerVelocityStats[p_controllerId].m_desiredVelocity = desiredV;
	// Location
	m_controllerLocationStats[p_controllerId].m_worldPos = pos;
	m_controllerLocationStats[p_controllerId].m_currentGroundPos = glm::vec3(pos.x, 0.0f, pos.z); // substitute this later with raycast to ground
}



void ControllerSystem::initControllerLocationAndVelocityStat(unsigned int p_idx, const glm::vec3& p_gaitGoalVelocity)
{
	glm::vec3 pos = getControllerPosition(p_idx);
	VelocityStat vstat(pos, p_gaitGoalVelocity);
	m_controllerVelocityStats.push_back(vstat);	
	LocationStat lstat{ pos, glm::vec3(pos.x,0.0f,pos.z) }; // In future, use raycast for ground pos
	m_controllerLocationStats.push_back(lstat);
}


glm::vec3 ControllerSystem::getControllerPosition(unsigned int p_controllerId)
{
	ControllerComponent* controller = m_controllers[p_controllerId];
	return getControllerPosition(controller);
}

glm::vec3 ControllerSystem::getControllerPosition(ControllerComponent* p_controller)
{
	return getLegFramePosition(p_controller->getLegFrame(0));
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

void ControllerSystem::updateFeet( unsigned int p_controllerId, ControllerComponent* p_controller )
{
	for (unsigned int i = 0; i < p_controller->getLegFrameCount(); i++)
	{
		ControllerComponent::LegFrame* lf = p_controller->getLegFrame(i);
		unsigned int legCount = (unsigned int)lf->m_legs.size();
		glm::vec3 currentV = m_controllerVelocityStats[p_controllerId].m_currentVelocity;
		glm::vec3 desiredV = m_controllerVelocityStats[p_controllerId].m_desiredVelocity;
		glm::vec3 goalV = m_controllerVelocityStats[p_controllerId].m_goalVelocity;
		glm::vec3 groundPos = m_controllerLocationStats[p_controllerId].m_currentGroundPos;
		float phi = p_controller->m_player.getPhase();
		for (unsigned int i = 0; i < legCount; i++)
		{
			//updateReferenceFootPosition(phi, m_runTime, goalV);
			updateFoot(p_controllerId, lf, i, phi, currentV, desiredV, groundPos);
			// !!!!!!!! m_legFrames[i].tempApplyFootTorque(m_player.m_gaitPhase); !!!!!!!!!!!SHOULD BE DONE WITH PD!!!!!!!!!
		}
	}
}

// Calculate the next position where the foot should be placed for legs in swing
void ControllerSystem::updateFoot(unsigned int p_controllerId, ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi, 
	const glm::vec3& p_velocity, const glm::vec3& p_desiredVelocity, const glm::vec3& p_groundPos)
{
	// The position is updated as long as the leg
	// is in stance. This means that the last calculated
	// position when the foot leaves the ground is used.
	if (isInControlledStance(p_lf, p_legIdx, p_phi))
	{
		updateFootStrikePosition(p_controllerId, p_lf, p_legIdx, p_phi, p_velocity, p_desiredVelocity, p_groundPos);
		offsetFootTargetDownOnLateStrike(p_lf, p_legIdx);
	}
	else // If the foot is in swing instead, start updating the current foot swing target
	{    // along the appropriate trajectory.
		updateFootSwingPosition(p_lf, p_legIdx, p_phi);
	}
}

// Calculate a new position where to place a foot.
// This is where the foot will try to swing to in its
// trajectory.
void ControllerSystem::updateFootStrikePosition(unsigned int p_controllerId, ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi, 
	const glm::vec3& p_velocity, const glm::vec3& p_desiredVelocity, const glm::vec3& p_groundPos)
{
	// Set the lift position to the old strike position (used for trajectory planning)
	// the first time each cycle that we enter this function
	if (!p_lf->m_footLiftPlacementPerformed[p_legIdx])
	{
		p_lf->m_footLiftPlacement[p_legIdx] = p_lf->m_footStrikePlacement[p_legIdx];
		p_lf->m_footLiftPlacementPerformed[p_legIdx] = true;
	}
	// Calculate the new position
	float mirror = (float)(p_legIdx) * 2.0f - 1.0f; // flips the coronal axis for the left leg
	glm::vec3 stepDir(mirror * p_lf->m_stepLength.x, 0.0f, p_lf->m_stepLength.y);
	glm::vec3 regularFootPos = MathHelp::transformDirection(getDesiredWorldOrientation(p_controllerId)/*getLegFrameTransform(p_lf)*/, stepDir);
	glm::vec3 lfPos = getLegFramePosition(p_lf);
	glm::vec3 finalPos = lfPos + calculateVelocityScaledFootPos(p_lf, regularFootPos, p_velocity, p_desiredVelocity);
	glm::vec3* b = &p_lf->m_footStrikePlacement[p_legIdx];
	p_lf->m_footStrikePlacement[p_legIdx] = projectFootPosToGround(finalPos, p_groundPos);
}

void ControllerSystem::updateFootSwingPosition(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi)
{
	//Vector3 oldPos = m_footTarget[p_idx]; // only for debug...
	//
	p_lf->m_footLiftPlacementPerformed[p_legIdx] = false; // reset
	// Get the fractional swing phase
	float swingPhi = p_lf->m_stepCycles[p_legIdx].getSwingPhase(p_phi);
	// The height offset, ie. the "lift" that the foot makes between stepping points.
	glm::vec3 heightOffset(0.0f, p_lf->m_stepHeighTraj.lerpGet(swingPhi), 0.0f);
	//m_currentFootGraphHeight[p_idx] = heightOffset.y;
	// scale the phi based on the easing function, for ground plane movement
	swingPhi = getFootTransitionPhase(p_lf,swingPhi);
	// Calculate the position
	// Foot movement along the ground
	glm::vec3 groundPlacement = glm::lerp(p_lf->m_footLiftPlacement[p_legIdx], p_lf->m_footStrikePlacement[p_legIdx], swingPhi);
	p_lf->m_footTarget[p_legIdx] = groundPlacement + heightOffset;
	//
// 	Color dbg = Color.green;
// 	if (p_idx == 0)
// 		dbg = Color.red;
// 	Debug.DrawLine(oldPos, m_footTarget[p_idx], dbg, 1.0f);
}

void ControllerSystem::offsetFootTargetDownOnLateStrike(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx)
{
	bool isTouchingGround = isFootStrike(p_lf,p_legIdx);
	if (!isTouchingGround)
	{
		glm::vec3 old = p_lf->m_footStrikePlacement[p_legIdx];
		p_lf->m_footStrikePlacement[p_legIdx] += glm::vec3(0.0f, -1.0f, 0.0f) * p_lf->m_lateStrikeOffsetDeltaH;
		dbgDrawer()->drawLine(old, p_lf->m_footStrikePlacement[p_legIdx], dawnBringerPalRGB[COL_NAVALBLUE], dawnBringerPalRGB[COL_ORANGE]);
	}
}


// Project a foot position to the ground beneath it
glm::vec3 ControllerSystem::projectFootPosToGround(const glm::vec3& p_footPosLF, const glm::vec3& p_groundPos) const
{
	return glm::vec3(p_footPosLF.x, p_groundPos.y, p_footPosLF.z);
}

// Scale a foot strike position prediction to the velocity difference
glm::vec3 ControllerSystem::calculateVelocityScaledFootPos(const ControllerComponent::LegFrame* p_lf, const glm::vec3& p_footPosLF,
	const glm::vec3& p_velocity,const glm::vec3& p_desiredVelocity) const
{
	return p_footPosLF + (p_velocity - p_desiredVelocity) * p_lf->m_footPlacementVelocityScale;
}

// Get the phase value in the foot transition based on
// swing phase. Note the phi variable here is the fraction
// of the swing phase!
float ControllerSystem::getFootTransitionPhase(ControllerComponent::LegFrame* p_lf, float p_swingPhi)
{
	return p_lf->m_footTransitionEase.lerpGet(p_swingPhi);
}




void ControllerSystem::updateTorques(unsigned int p_controllerId, ControllerComponent* p_controller, float p_dt)
{
	float phi = p_controller->m_player.getPhase();
	unsigned int torqueCount = p_controller->getTorqueListChunkSize();
	unsigned int torqueIdxOffset = p_controller->getTorqueListOffset();
	//// Get the three variants of torque
	//Vector3[] tPD = computePDTorques(phi);
	//Vector3[] tCGVF = computeCGVFTorques(phi, p_dt);
	//std::vector<glm::vec3> tPD(torqueCount);
	////std::vector<glm::vec3> tCGVF(torqueCount);
	//std::vector<glm::vec3> tVF(torqueCount);
	//for (unsigned int i = 0; i < torqueCount; i++)
	//{
	//	tPD[i] = glm::vec3(0.0f); /*tCGVF[i] = glm::vec3(0.0f);*/ tVF[i] = glm::vec3(0.0f);
	//}
	//
	computePDTorques(&m_jointTorques, p_controller, p_controllerId, torqueIdxOffset, phi, p_dt);
	computeAllVFTorques(&m_jointTorques, p_controller, p_controllerId, torqueIdxOffset, phi, p_dt);
	////// Sum them (Right now, we're writing directly to the global array
	// Summing of partial lists might be good if we parallelize this step as well
	//for (unsigned int i = 0; i < torqueCount; i++)
	//{
	//	m_jointTorques[torqueIdxOffset + i] = /*tPD[i] + */tVF[i];
	//		//= glm::vec3(0.0f, 200.0f, 0.0f);
	//		//+
	//}
	//
	// Apply them to the leg frames, also
	// feed back corrections for hip joints
	for (unsigned int i = 0; i < p_controller->getLegFrameCount(); i++)
	{
		applyNetLegFrameTorque(p_controllerId, p_controller, i, phi, p_dt);
	}
}

void ControllerSystem::calculateLegFrameNetLegVF(unsigned int p_controllerIdx, ControllerComponent::LegFrame* p_lf, float p_phi, float p_dt, 
										 VelocityStat& p_velocityStats)
{
	unsigned int legCount = (unsigned int)p_lf->m_legs.size(),
				 stanceLegs = 0;
	bool* legInStance = new bool[legCount];
	
	// First we need to count the stance legs
	for (unsigned int i = 0; i < legCount; i++)
	{
		legInStance[i] = false;
		if (isInControlledStance(p_lf,i, p_phi))
		{
			stanceLegs++; legInStance[i] = true;
		}
	}
	// 
	// Declare stance forces here as they should only be 
	// calculated once per leg frame, then reused
	glm::vec3 fv(0.0f), fh(0.0f); bool stanceForcesCalculated = false;
	// Run again and calculate forces
	for (unsigned int i = 0; i < legCount; i++)
	{
		ControllerComponent::Leg* leg = &p_lf->m_legs[i];
		// for this list, same for all:
		unsigned int vfIdx = leg->m_DOFChain.m_vfIdxList[0];
		glm::vec3 dbgFootPos = getFootPos(p_lf, i);
		// Swing force
		if (!legInStance[i])
		{
			glm::vec3 fsw(calculateFsw(p_lf, i, p_phi, p_dt));
			m_VFs[vfIdx] = calculateSwingLegVF(fsw); // Store force
			//
			dbgDrawer()->drawLine(dbgFootPos, dbgFootPos + m_VFs[vfIdx], dawnBringerPalRGB[COL_PINK], dawnBringerPalRGB[COL_PURPLE]);
		}
		else
			// Stance force
		{
			if (!stanceForcesCalculated)
			{
				fv = calculateFv(p_lf, m_controllerVelocityStats[p_controllerIdx]);
				fh = calculateFh(p_lf, m_controllerLocationStats[p_controllerIdx], p_phi, p_dt, glm::vec3(0.0f, 1.0f, 0.0));
				stanceForcesCalculated=true;
			}	
			glm::vec3 fd(calculateFd(p_controllerIdx,p_lf, i));
			m_VFs[vfIdx] = calculateStanceLegVF(stanceLegs, fv, fh, fd); // Store force
			dbgDrawer()->drawLine(dbgFootPos, dbgFootPos + m_VFs[vfIdx], dawnBringerPalRGB[COL_LIGHTBLUE], dawnBringerPalRGB[COL_NAVALBLUE]);
		}
		// Debug test
		//leg->m_DOFChain.vf = glm::vec3(0.0f, 50.0f*sin((float)m_runTime*0.2f), 0.0f);
	}

	delete[] legInStance;
}

void ControllerSystem::computeAllVFTorques(std::vector<glm::vec3>* p_outTVF, ControllerComponent* p_controller, 
	unsigned int p_controllerIdx, unsigned int p_torqueIdxOffset, float p_phi, float p_dt)
{
	if (m_useVFTorque)
	{
		for (unsigned int i = 0; i < p_controller->getLegFrameCount(); i++)
		{
			ControllerComponent::LegFrame* lf = p_controller->getLegFrame(i);
			calculateLegFrameNetLegVF(i, lf, p_phi, p_dt, m_controllerVelocityStats[p_controllerIdx]);
			// Begin calculating Jacobian transpose for each leg in leg frame
			unsigned int legCount = (unsigned int)lf->m_legs.size();	
			for (unsigned int n = 0; n < legCount; n++)
			{
				computeVFTorquesFromChain(p_outTVF, lf, n, ControllerComponent::STANDARD_CHAIN, p_torqueIdxOffset, p_phi, p_dt);
				if (isInControlledStance(lf, n, p_phi))
					computeVFTorquesFromChain(p_outTVF, lf, n, ControllerComponent::GRAVITY_COMPENSATION_CHAIN, p_torqueIdxOffset, p_phi, p_dt);
			}
		}
	}
}

void ControllerSystem::computeVFTorquesFromChain(std::vector<glm::vec3>* p_outTVF, ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx,
	ControllerComponent::VFChainType p_type, unsigned int p_torqueIdxOffset, float p_phi, float p_dt)
{
	// calculating Jacobian transpose for specified leg in leg frame
	// Calculate torques using specified leg chain
	ControllerComponent::Leg* leg = &p_lf->m_legs[p_legIdx];
	ControllerComponent::VFChain* chain = leg->getVFChain(p_type);
	// Get the end effector position
	// We're using the COM of the foot
	glm::vec3 end = getFootPos(p_lf, p_legIdx);
	// Calculate the matrices
	CMatrix J = JacobianHelper::calculateVFChainJacobian(*chain,						// Chain of DOFs to solve for
		end,						// Our end effector goal position
		&m_VFs,							// All virtual forces
		&m_jointWorldInnerEndpoints,	// All joint rotational axes
		&m_jointWorldTransforms);		// All joint world transformations
	CMatrix Jt = CMatrix::transpose(J);

	glm::mat4 sum(0.0f);
	for (unsigned int g = 0; g < m_jointWorldInnerEndpoints.size(); g++)
	{
		sum += m_jointWorldTransforms[g];
	}
	DEBUGPRINT(((std::string("\n") + std::string(" WTransforms: ") + ToString(sum)).c_str()));
	DEBUGPRINT(((std::string("\n") + std::string(" Pos: ") + ToString(end)).c_str()));
	//DEBUGPRINT(((std::string("\n") + std::string(" VF: ") + ToString(vf)).c_str()));

	// Use matrix to calculate and store torque
	for (unsigned int m = 0; m < chain->getSize(); m++)
	{
		// store torque
		unsigned int localJointIdx = chain->m_jointIdxChain[m];
		glm::vec3 vf = m_VFs[chain->m_vfIdxList[m]];
		glm::vec3 JjVec(J(0, m), J(1, m), J(2, m));
		glm::vec3 JVec(Jt(m, 0), Jt(m, 1), Jt(m, 2));
		glm::vec3 addT = (chain->m_DOFChain)[m] * glm::dot(JVec, vf);
		//
		float ssum = JjVec.x + JjVec.y + JjVec.z;
		DEBUGPRINT(((std::string("\n") + ToString(m) + std::string(" J sum: ") + ToString(ssum)).c_str()));
		ssum = JVec.x + JVec.y + JVec.z;
		DEBUGPRINT(((std::string("\n") + ToString(m) + std::string(" Jt sum: ") + ToString(ssum)).c_str()));
		(*p_outTVF)[localJointIdx + p_torqueIdxOffset] += addT; // Here we could write to the global list instead directly maybe as an optimization
		// Do it like this for now, for the sake of readability and debugging.
	}
}

bool ControllerSystem::isInControlledStance(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi)
{
	// Check if in stance and also read as stance if the 
	// foot is really touching the ground while in end of swing
	StepCycle* stepCycle = &p_lf->m_stepCycles[p_legIdx];
	bool stance = stepCycle->isInStance(p_phi);
	if (!stance)
	{
		bool isTouchingGround = isFootStrike(p_lf,p_legIdx);
		if (isTouchingGround)
		{
			float swing = stepCycle->getSwingPhase(p_phi);
			if (swing > 0.8f) // late swing as defined and mentioned by coros et al (quadruped locomotion)
			{
				// update foot target to it's final so it's not stuck in air
				updateFootSwingPosition(p_lf, p_legIdx, 1.0f); 
				stance = true;
			}
		}
	}
	return stance;
}

glm::vec3 ControllerSystem::calculateFsw(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi, float p_dt)
{
	float swing = p_lf->m_stepCycles[p_legIdx].getSwingPhase(p_phi);
	float Kft = p_lf->m_footTrackingGainKp.lerpGet(swing);
	p_lf->m_footTrackingSpringDamper.setKp_KdEQTenPrcntKp(Kft);
	glm::vec3 diff = getFootPos(p_lf,p_legIdx) - p_lf->m_footStrikePlacement[p_legIdx];
	float error = glm::length(diff);
	return -glm::normalize(diff) * p_lf->m_footTrackingSpringDamper.drive(error, p_dt);
}




glm::vec3 ControllerSystem::calculateFv(ControllerComponent::LegFrame* p_lf, const VelocityStat& p_velocityStats)
{
	return p_lf->m_velocityRegulatorKv*(p_velocityStats.m_desiredVelocity - p_velocityStats.m_currentVelocity);
}

glm::vec3 ControllerSystem::calculateFh(ControllerComponent::LegFrame* p_lf, const LocationStat& p_locationStat, float p_phi, float p_dt, const glm::vec3& p_up)
{
	float hLF = p_lf->m_heightLFTraj.lerpGet(p_phi);
	glm::vec3 currentHeight = p_locationStat.m_worldPos - p_locationStat.m_currentGroundPos;
	// the current height y only works for up=0,1,0
	// so in case we are making a space game, i'd reckon we should have the following PD work on vec3's
	// but for now, a float is OK
	return p_up * p_lf->m_FhPD.drive(hLF - currentHeight.y, p_dt); // PD
}

glm::vec3 ControllerSystem::calculateFd(unsigned int p_controllerId, ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx)
{
	/*
	velocity according to Fv = kv(vd-v). Third, a leg-specific virtual force FD(D) implements a phase-dependent force that is
	customised for each stance leg. This allows for modeling the indi-vidualized role of each leg in gaits such as the dog gallop [Walter
	and Carrier 2007]. Here, D measures forward progress in the gait
	and is computed as the ground-plane projection of PLF - Pfoot, where PLF is the location of the origin of the leg frame, and Pfoot
	and is computed as the ground-plane projection of PLF - Pfoot, where PLF is the location of the origin of the leg frame, and Pfoot
	is the location of the foot of a given leg. FD(D)
	*/
	glm::vec3 FD;
	// Check van de panne's answer before implementing this
	glm::vec3 Dvec = getLegFramePosition(p_lf) - getFootPos(p_lf,p_legIdx)/*-transform.position)*/;		
	// Project onto ground plance
	Dvec = projectFootPosToGround(Dvec, m_controllerLocationStats[p_controllerId].m_currentGroundPos);
	// transform to local space so we can interpret the terms as horizontal and vertical
	Dvec = MathHelp::invTransformDirection(getDesiredWorldOrientation(p_controllerId)/*getLegFrameTransform(p_lf)*/, Dvec);

	Dvec.x = 0.0f;
	glm::vec4 c = p_lf->m_FDHVComponents;

	/*[...] this is just during stance. I believe that each of the horizontal and vertical components of FD
			is defined to be a linear function of D, i.e.,
			FD_h = c0 + c1*D
			FD_v = c2+ c3*D
			So that is four parameters, per leg x 4 legs = 16 parameters.
			There is a discrepancy between this and the supplemental material,
			which lists this as only having 8 parameters  [...] 
			*/
	float D = Dvec.z;

	float FDhoriz = c[0] + c[1] * D/*Dvec.y*/;
	float FDvert =	c[2] + c[3] * D/*Dvec.z*/;
	
	//float FDx = m_tuneFD[p_legId, Dx].x;
	//float FDz = m_tuneFD[p_legId, Dz].z;
	////Debug.DrawLine(transform.position, transform.position + new Vector3(FDx, 0.0f, FDz), Color.magenta,1.0f);
	//// Transform back to world space so that it is applicable to current orientation
	//FD = MathHelp::transformDirection(getLegFrameTransform(p_lf), glm::vec3(0.0f, FDvert, FDhoriz));

	FD = MathHelp::transformDirection(getDesiredWorldOrientation(p_controllerId), glm::vec3(0.0f, FDvert, FDhoriz)); // try using wanted orientation, instead of the actual
	dbgDrawer()->drawLine(getFootPos(p_lf, p_legIdx), getFootPos(p_lf, p_legIdx) + FD*2.0f, dawnBringerPalRGB[COL_YELLOW], dawnBringerPalRGB[COL_ORANGE]);
	//FD = glm::vec3(0.0f);
	return FD;
}

glm::vec3 ControllerSystem::calculateSwingLegVF(const glm::vec3& p_fsw)
{
	return p_fsw; // Right now, this force is equivalent to fsw
}

glm::vec3 ControllerSystem::calculateStanceLegVF(unsigned int p_stanceLegCount,
	const glm::vec3& p_fv, const glm::vec3& p_fh, const glm::vec3& p_fd)
{
	float n = (float)p_stanceLegCount;
	return -p_fd - (p_fh / n) - (p_fv / n); // note fd should be stance fd
}


void ControllerSystem::applyNetLegFrameTorque(unsigned int p_controllerId, ControllerComponent* p_controller, unsigned int p_legFrameIdx, float p_phi, float p_dt)
{
	// Preparations, get a hold of all legs in stance,
	// all legs in swing. And get a hold of their and the 
	// closest spine's torques.
	ControllerComponent::LegFrame* lf = p_controller->getLegFrame(p_legFrameIdx);
	unsigned int legCount = (unsigned int)lf->m_legs.size();	
	unsigned int lfJointIdx = lf->m_legFrameJointId;
	glm::vec3 tstance(0.0f), tswing(0.0f), tspine(0.0f);
	unsigned int stanceCount = 0;
	unsigned int* stanceLegBuf = new unsigned int[legCount];
	for (unsigned int i = 0; i < legCount; i++)
	{
		unsigned int jointId = lf->m_hipJointId[i];
		if (isInControlledStance(lf, i, p_phi))
		{
			tstance += m_jointTorques[jointId];
			stanceLegBuf[stanceCount] = jointId;
			stanceCount++;
		}
		else
			tswing += m_jointTorques[jointId];
	}

	// Spine if it exists
	int spineIdx = (int)lf->m_spineJointId;
	if (spineIdx >= 0)
		tspine = m_jointTorques[(unsigned int)spineIdx];

	// 1. Calculate current torque for leg frame:
	// tLF = tstance + tswing + tspine.
	// Here the desired torque is feedbacked through the
	// stance legs (see 3) as their current torque
	// is the product of previous desired torque combined
	// with current real-world scenarios.
	glm::vec3 tLF = tstance + tswing + tspine;
	m_jointTorques[lfJointIdx] = tLF;

	// 2. Calculate a desired torque, tdLF, using the previous current
	// torque, tLF, and a PD-controller driving towards the 
	// desired orientation, omegaLF.
	glm::mat4 dWM = getDesiredWorldOrientation(p_controllerId);
	glm::vec3 hgr = MathHelp::transformDirection(dWM, glm::vec3(5.0f, 0.0f, 0.0f));
	glm::vec3 fram = MathHelp::transformDirection(dWM, glm::vec3(0.0f, 0.0f, 5.0f));
	//glm::mat4 up = glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0), glm::vec3(0.0f, 1.0f, 0.0));
	glm::quat desiredW = glm::quat_cast(dWM/**up*/);
	glm::quat omegaLF = lf->getCurrentDesiredOrientation(p_phi);
	omegaLF = desiredW*omegaLF;
	glm::quat currentOrientation = MathHelp::getMatrixRotation(m_jointWorldTransforms[lf->m_legFrameJointId]);
	glm::mat4 orient = glm::mat4_cast(currentOrientation);
	glm::vec3 tdLF = lf->getOrientationPDTorque(currentOrientation, omegaLF, p_dt);

	glm::vec3 chgr = MathHelp::transformDirection(orient, glm::vec3(5.0f, 0.0f, 0.0f));
	glm::vec3 cfram = MathHelp::transformDirection(orient, glm::vec3(0.0f, 0.0f, 5.0f));

	// Draw ORIENTATION distance
	glm::vec3 wanted = MathHelp::transformDirection(glm::mat4_cast(omegaLF), glm::vec3(0.0f, 10.0f, 0.0f));
	glm::vec3 current = MathHelp::transformDirection(glm::mat4_cast(currentOrientation), glm::vec3(0.0f, 10.0f, 0.0f));
	dbgDrawer()->drawLine(getLegFramePosition(lf), getLegFramePosition(lf) + wanted, dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
	dbgDrawer()->drawLine(getLegFramePosition(lf), getLegFramePosition(lf) + current, dawnBringerPalRGB[COL_MOSSGREEN], dawnBringerPalRGB[COL_MOSSGREEN]);
	dbgDrawer()->drawLine(getLegFramePosition(lf) + current, getLegFramePosition(lf) + wanted, dawnBringerPalRGB[COL_MOSSGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
	dbgDrawer()->drawLine(getLegFramePosition(lf), getLegFramePosition(lf) + hgr, dawnBringerPalRGB[COL_RED], dawnBringerPalRGB[COL_RED]);
	dbgDrawer()->drawLine(getLegFramePosition(lf), getLegFramePosition(lf) + fram, dawnBringerPalRGB[COL_LIGHTBLUE], dawnBringerPalRGB[COL_LIGHTBLUE]);
	dbgDrawer()->drawLine(getLegFramePosition(lf), getLegFramePosition(lf) + chgr, dawnBringerPalRGB[COL_RED] * 0.8f, dawnBringerPalRGB[COL_RED] * 0.8f);
	dbgDrawer()->drawLine(getLegFramePosition(lf), getLegFramePosition(lf) + cfram, dawnBringerPalRGB[COL_LIGHTBLUE] * 0.8f, dawnBringerPalRGB[COL_LIGHTBLUE] * 0.8f);
	// test code
	//rigidbody.AddTorque(tdLF);

	// 3. Now loop through all legs in stance (N) and
	// modify their torques in the vector according
	// to tstancei = (tdLF-tswing-tspine)/N
	// This is to try to make the stance legs compensate
	// for current errors in order to make the leg frame
	// reach its desired torque.
	int N = stanceCount;
	for (int i = 0; i < N; i++)
	{
		unsigned int idx = stanceLegBuf[i];
		m_jointTorques[idx] = (tdLF - tswing - tspine) / (float)N;
	}
	delete[] stanceLegBuf;
	// The vector reference to the torques, now contains the new LF torque
	// as well as any corrected stance-leg torques.
}



void ControllerSystem::computePDTorques(std::vector<glm::vec3>* p_outTVF, ControllerComponent* p_controller, unsigned int p_controllerIdx, unsigned int p_torqueIdxOffset, float p_phi, float p_dt)
{
	throw std::exception("The method or operation is not implemented.");
}



glm::vec3 ControllerSystem::getFootPos( ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx )
{
	return MathHelp::getMatrixTranslation(m_jointWorldTransforms[p_lf->m_feetJointId[p_legIdx]]);
}

glm::mat4& ControllerSystem::getLegFrameTransform(const ControllerComponent::LegFrame* p_lf)
{
	return m_jointWorldTransforms[p_lf->m_legFrameJointId];
}

glm::vec3 ControllerSystem::getLegFramePosition(const ControllerComponent::LegFrame* p_lf) const
{
	unsigned int legFrameJointId = p_lf->m_legFrameJointId;
	return MathHelp::getMatrixTranslation(m_jointWorldTransforms[legFrameJointId]);
}

// Get the desired world space orientation based on the current desired velocity
glm::mat4 ControllerSystem::getDesiredWorldOrientation(unsigned int p_controllerId) const
{
	glm::vec3 dir = m_controllerVelocityStats[p_controllerId].m_desiredVelocity;
	dir.z *= -1.0f; // as we're using a ogl function below, we flip z
	return glm::lookAt(glm::vec3(0.0f), dir, glm::vec3(0.0f, 1.0f, 0.0f));
}

bool ControllerSystem::isFootStrike(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx)
{
	return p_lf->m_footIsColliding[p_legIdx];
}

// Write the collision status of feet to array
// Has to be done before all other controller logic, as pre process
// Coupling to artemis and rigidbody component :/
void ControllerSystem::writeFeetCollisionStatus(ControllerComponent* p_controller)
{
	for (unsigned int i = 0; i < p_controller->getLegFrameCount(); i++)
	{
		ControllerComponent::LegFrame* lf = p_controller->getLegFrame(i);
		unsigned int legCount = (unsigned int)lf->m_legs.size();
		for (unsigned int i = 0; i < legCount; i++)
		{
			unsigned int footRBIdx = lf->m_footRigidBodyIdx[i];
			lf->m_footIsColliding[i] = m_footRigidBodyRefs[footRBIdx]->isColliding();
		}
	}
	// phew
}
