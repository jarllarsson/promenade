#include "ControllerSystem.h"

//#include <ppl.h>
#include <omp.h>
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

bool ControllerSystem::m_useVFTorque=true;
bool ControllerSystem::m_useGCVFTorque=true;
bool ControllerSystem::m_usePDTorque=true;
bool ControllerSystem::m_useLFFeedbackTorque = true;
bool ControllerSystem::m_dbgShowVFVectors = true;
bool ControllerSystem::m_dbgShowGCVFVectors = true;
bool ControllerSystem::m_dbgShowTAxes = true;
float ControllerSystem::m_torqueLim = 100.0f;


ControllerSystem::~ControllerSystem()
{

}



void ControllerSystem::removed(artemis::Entity &e)
{
	//DEBUGPRINT(("\nREMOVING CONTROLLER\n"));
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
		for (int x = 0; x < controller->getLegFrameCount();x++)
		{
			ControllerComponent::LegFrame* lf = controller->getLegFrame(x);
			unsigned int legCount = (unsigned int)lf->m_legs.size();
			for (unsigned int i = 0; i < legCount; i++)
			{
				unsigned int jointId = lf->m_hipJointId[i];
				if (isInControlledStance(lf, i, controller->m_player.getPhase()))
				{
					for (int n = 0; n < 3; n++) // 3 segments
					{
						artemis::Entity* segEntity = m_dbgJointEntities[jointId + n];
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
}

void ControllerSystem::fixedUpdate(float p_dt)
{
	m_runTime += p_dt;
	m_steps++;

	double startTiming = Time::getTimeMs();
	double startTimingOmp = omp_get_wtime();
	// Clear debug draw batch 
	// (not optimal to only do it here if drawing from game systems,
	// batch calls should be put in a map or equivalent)
	dbgDrawer()->clearDrawCalls();

	// Update all transforms
	for (unsigned int i = 0; i < m_jointRigidBodies.size(); i++)
	{
		saveJointMatrix(i);
		m_oldJointTorques[i] = m_jointTorques[i];
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

		if (m_executionSetup==SERIAL)
		{
			// =====================================
			// Single threaded implementation
			// =====================================
			for (int n = 0; n < controllerCount; n++)
			{
				ControllerComponent* controller = m_controllers[(unsigned int)n];
				// Run controller code here
				controllerUpdate((unsigned int)n, p_dt);
			}
		}
		else
		{
			// =====================================
			// Multi threaded CPU implementation
			// =====================================
			dbgDrawer()->m_enabled = false;
			int loopInvoc = 4;
			int serialChars = 20;
			/*concurrency::parallel_for(0, loopInvoc, [&](int n)
			{
			for (int i = 0; i < serialChars; i++)
			{
			// character id is indexed from serial- and parallel invoc
			int id = i + (n*serialChars);
			if (id<controllerCount)
			{
			ControllerComponent* controller = m_controllers[id];
			// Run controller code here
			controllerUpdate(id, p_dt);
			}
			}
			});*/
			#pragma omp parallel num_threads(8)
			{
				int n = omp_get_thread_num();
				int test = 0;
				for (int i = 0; i < serialChars; i++)
				{
					//test++;
					// character id is indexed from serial- and parallel invoc
					int id = i + (n*serialChars);
					if (id < controllerCount)
					{
						ControllerComponent* controller = m_controllers[id];
						// Run controller code here
						controllerUpdate(id, p_dt);
					}
				}
			}
		}

	}
	else
	{
		DEBUGPRINT(("\nNO CONTROLLERS YET\n"));
	}
	double endTimingOmp = omp_get_wtime();
	m_timing = Time::getTimeMs() - startTiming;
	//m_timing = endTimingOmp - startTimingOmp;
	if (m_perfRecorder != NULL)
		m_perfRecorder->saveMeasurement(m_timing,m_steps);
}

void ControllerSystem::finish()
{

}

void ControllerSystem::applyTorques( float p_dt )
{
	if (m_jointRigidBodies.size() == m_jointTorques.size())
	{
		float tLim = m_torqueLim;
		for (unsigned int i = 0; i < m_jointRigidBodies.size(); i++)
		{
			glm::vec3 t = m_jointTorques[i];
			if (glm::length(t)>tLim) 
				t = glm::normalize(t)*tLim;
			if (m_dbgShowTAxes && glm::length(t) > 0)
			{
				glm::vec3 pos = getJointPos(i);
				dbgDrawer()->drawLine(pos, pos + t, dawnBringerPalRGB[COL_LIGHTBLUE], dawnBringerPalRGB[COL_LIGHTBLUE]);
 			}
			// The torque being applied is:
			// clockwise along x axis = +x
			// clockwise along y axis = +y
			// clockwise along z axis = +z
			// Where clockwise along an axis is when watching the rotation at the "top" of the axis down towards origo.
			// Counter-clockwise is then the opposite (-x,-y and -z)
			m_jointRigidBodies[i]->applyTorque(btVector3(t.x, t.y, t.z));
		}
	}
}

void ControllerSystem::buildCheck()
{
	for (unsigned int i = 0; i < m_controllersToBuild.size(); i++) // controllers
	{
		glm::vec3 startGaitVelocity(0.0f, 0.0f, 0.5f);
		ControllerComponent* controller = m_controllersToBuild[i];
		// start by storing the current torque list size as offset, this'll be where we'll begin this
		// controller's chunk of the torque list
		unsigned int torqueListOffset = (unsigned int)m_jointTorques.size();
		// LEG FRAMES
		for (int n = 0; n < controller->getLegFrameCount(); n++) // leg frames
		{
			ControllerComponent::LegFrameEntityConstruct* legFrameEntities = controller->getLegFrameEntityConstruct(n);
			ControllerComponent::LegFrame* legFrame = controller->getLegFrame(n);
			// Build the controller (Temporary code)
			// The below should be done for each leg (even the root)
			// Create ROOT
			RigidBodyComponent* rootRB = (RigidBodyComponent*)legFrameEntities->m_legFrameEntity->getComponent<RigidBodyComponent>();
			TransformComponent* rootTransform = (TransformComponent*)legFrameEntities->m_legFrameEntity->getComponent<TransformComponent>();
			unsigned int rootIdx = addJoint(rootRB, rootTransform, controller);
			m_rigidBodyRefs.push_back(rootRB);
			m_dbgJointEntities.push_back(legFrameEntities->m_legFrameEntity); // for easy debugging options
			//
			legFrame->m_legFrameJointId = rootIdx; // store idx to root for leg frame
			legFrame->m_startPosOffset = rootTransform->getPosition();
			// prepare legs			
			unsigned int legCount = legFrameEntities->m_upperLegEntities.size();
			legFrame->m_legs.resize(legCount); // Allocate the number of specified legs
			// when and if we have a spine, add it here and store its id to leg frame
			// Might have to do this after all leg frames instead, due to the link being at different ends of spine
			legFrame->m_spineJointId = -1; // -1 means it doesn't exist
			//
			glm::vec3 legFramePos = rootTransform->getPosition(), footPos;
			unsigned int footJointId = 0;
#pragma region legs
			for (unsigned int x = 0; x < legCount; x++)
			{
				// ROOT
				// ---------------------------------------------
				m_VFs.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
				unsigned int vfIdx = (unsigned int)((int)m_VFs.size() - 1);
				ControllerComponent::VFChain* standardDOFChain = legFrame->m_legs[x].getVFChain(ControllerComponent::VFChainType::STANDARD_CHAIN);
				// start by adding the already existing root id (needed in all leg chains)
				addJointToVFChain(standardDOFChain, rootIdx, vfIdx);
				// LEGS
				// ---------------------------------------------
				// Traverse the segment structure for the leg to get the rest
				artemis::Entity* jointEntity = legFrameEntities->m_upperLegEntities[x];
				int jointsAddedForLeg = 0;
				int kneeFlip = 1;
				while (jointEntity != NULL) // read all joints
				{
					// Get joint data
					TransformComponent* jointTransform = (TransformComponent*)jointEntity->getComponent<TransformComponent>();
					RigidBodyComponent* jointRB = (RigidBodyComponent*)jointEntity->getComponent<RigidBodyComponent>();
					ConstraintComponent* parentLink = (ConstraintComponent*)jointEntity->getComponent<ConstraintComponent>();
					// Add the joint
					unsigned int idx = addJoint(jointRB, jointTransform, controller);
					m_rigidBodyRefs.push_back(jointRB);
					m_dbgJointEntities.push_back(jointEntity); // for easy debugging options
					// Get DOF on joint to chain
					addJointToVFChain(standardDOFChain, idx, vfIdx, parentLink->getDesc()->m_angularDOF_LULimits);
					// Register joint for PD (and create PD)
					float kp = 0.0f, kd = 0.0f;
					if (jointsAddedForLeg == 0)
					{
						kp = legFrame->m_ulegPDsK.x; kd = legFrame->m_ulegPDsK.y;// upper
					} 
					else if (jointsAddedForLeg < 2)
					{
						kp = legFrame->m_llegPDsK.x; kd = legFrame->m_llegPDsK.y;// lower
						if (parentLink->getUpperLim().x>PI*0.01f) // if we have a upper lim over 0, we have a "digitigrade knee"
							kneeFlip = -1; // and we need to reflect this for the IK solver
					} 
					else
					{
						kp = legFrame->m_flegPDsK.x; kd = legFrame->m_flegPDsK.y;// foot
					} 
					addJointToPDChain(legFrame->m_legs[x].getPDChain(), idx, kp, kd);
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
						legFrame->m_footHeight = jointTransform->getScale().z; // height between dorsum and sole(remember; foot is rotated limb)
						legFrame->createFootPlacementModelVarsForNewLeg(footPos);
						// add rigidbody idx so we can check for collisions and late foot strikes
						legFrame->m_footRigidBodyIdx.push_back(m_rigidBodyRefs.size() - 1);
						footJointId = idx;
						//
						jointEntity = NULL;
					}
					jointsAddedForLeg++;
				}
				// Copy all DOFs for ordinary VF-chain to the gravity compensation chain
				legFrame->m_legs[x].m_DOFChainGravityComp = legFrame->m_legs[x].m_DOFChain;
				unsigned int origGCDOFsz = legFrame->m_legs[x].m_DOFChainGravityComp.getSize();
				// Change the VFs for this list, as they need to be used to counter gravity
				// They're also static, so we only need to do this once
				int oldJointGCIdx = -1;
				vfIdx = 0;
				for (unsigned int m = 0; m < origGCDOFsz; m++)
				{
					unsigned int jointId = legFrame->m_legs[x].m_DOFChainGravityComp.m_jointIdxChain[m];
					if (jointId != oldJointGCIdx)
					{
						float mass = m_jointMass[jointId];
						m_VFs.push_back(-mass*glm::vec3(0.0f, WORLD_GRAVITY, 0.0f));
						vfIdx = (unsigned int)((int)m_VFs.size() - 1);
						legFrame->m_legs[x].m_DOFChainGravityComp.m_jointIdxChainOffsets.push_back(m);
					}
					legFrame->m_legs[x].m_DOFChainGravityComp.m_vfIdxList[m] = vfIdx;
					oldJointGCIdx = jointId;
				}
				// ANGLE TARGETS; IK- AND FOOT
				// ---------------------------------------------
				// Add an IK handler for leg
				legFrame->m_legIK.push_back(IK2Handler(kneeFlip));
				// add entry for foot rotation timing params in struct
				legFrame->m_toeOffTime.push_back(0.0f);
				legFrame->m_tuneFootStrikeTime.push_back(0.0f);
			}
#pragma endregion legs
			legFrame->m_height = legFramePos.y - (footPos.y - legFrame->m_footHeight*0.5f/*m_jointLengths[footJointId]*0.5f*/);
		}
		//
		// SPINE
		// ---------------------------------------------
		int spineJoints = controller->getSpineJointEntitiesConstructSize();
		controller->m_spine.m_joints = spineJoints;
		if (spineJoints>0)
		{
			float skp = controller->m_spine.m_PDsK.x, skd = controller->m_spine.m_PDsK.y;
			// begin with LF as base
			// TODO! Two chains, one with front LF as base
			// and the other with the back LF. The spine needs two "runs",
			// in order to use both LFs as bases. The consecutive forces are
			// thus weighted, as you can see below in the following pushes 
			// of constant-vectors to m_VFs
			// =======================================
			// The virtual force is shared across the front and rear leg frames with respective weights
			// of w and 1 - w, where w E[0, 1] is the fractional distance along the spine of the given link, 
			// with w = 1 at the front leg frame and w = 0 at the rear leg frame.
			float spineGC_w = 1.0f;
			ControllerComponent::LegFrame* lf = controller->getLegFrame(0);
			unsigned int rootIdx = lf->m_legFrameJointId;
			float rootmass = m_jointMass[rootIdx];
			m_VFs.push_back(-rootmass*glm::vec3(0.0f, WORLD_GRAVITY, 0.0f)*spineGC_w);
			unsigned int rootvfIdx = (unsigned int)((int)m_VFs.size() - 1);
			addJointToVFChain(controller->m_spine.getGCVFChainFwd(), rootIdx, rootvfIdx);
			// Now add back LF to back-chain (same procedure)
			lf = controller->getLegFrame(1); rootIdx = lf->m_legFrameJointId; rootmass = m_jointMass[rootIdx];
			m_VFs.push_back(-rootmass*glm::vec3(0.0f, WORLD_GRAVITY, 0.0f)*spineGC_w);
			rootvfIdx = (unsigned int)((int)m_VFs.size() - 1);
			addJointToVFChain(controller->m_spine.getGCVFChainBwd(), rootIdx, rootvfIdx);
			// then the rest of the spine joints (except the other end LF
			for (int si = 0; si < controller->getLegFrameCount();si++)
			{
				int start = 0, iter = 1, end = spineJoints;
				if (si > 0) { start = spineJoints - 1; iter = -1; end = -1; } // reverse on second
				for (int s = start; s != end; s+=iter) // read all joints TODO! right now this works, as we're constructing only one chain, 
					// but when we're making two later on, we must iterate backwards the second time, to get the correct order
				{
					artemis::Entity* jointEntity = controller->getSpineJointEntitiesConstruct(s);
					// Get joint data
					TransformComponent* jointTransform = (TransformComponent*)jointEntity->getComponent<TransformComponent>();
					RigidBodyComponent* jointRB = (RigidBodyComponent*)jointEntity->getComponent<RigidBodyComponent>();
					ConstraintComponent* parentLink = (ConstraintComponent*)jointEntity->getComponent<ConstraintComponent>();
					// Add the joint
					unsigned int idx = addJoint(jointRB, jointTransform, controller);
					m_rigidBodyRefs.push_back(jointRB);
					m_dbgJointEntities.push_back(jointEntity); // for easy debugging options
					// Get DOF on joint to GCVF chain, the spine does not use ordinary VFs, so we have to set up the base here
					float mass = m_jointMass[idx];
					if (si==0)
						spineGC_w = 1.0f - ((float)(s + 1) / (float)(spineJoints + 2)); // get the spine fractional distance, here we count the LFs as spine joints as well.
					else
						spineGC_w = 1.0f - ((float)(spineJoints - (s + 1)) / (float)(spineJoints + 2)); // backwards, still need "first" to have highest score
					// Thus we get start offset of 1(LF/root is spine 0), and a size of +2 (front- and back LF)
					m_VFs.push_back(-mass*glm::vec3(0.0f, WORLD_GRAVITY, 0.0f)*spineGC_w);
					unsigned int vfIdx = (unsigned int)((int)m_VFs.size() - 1);
					ControllerComponent::VFChain* vfchain = NULL;
					if (si == 0)
						vfchain = controller->m_spine.getGCVFChainFwd();
					else
						vfchain = controller->m_spine.getGCVFChainBwd();
					addJointToVFChain(vfchain, idx, vfIdx, parentLink->getDesc()->m_angularDOF_LULimits);
					// addJointToStandardVFChain(standardDOFChain, idx, vfIdx, parentLink->getDesc()->m_angularDOF_LULimits);
					// Register joint for PD (and create PD)
					if (si==0) addJointToPDChain(controller->m_spine.getPDChain(), idx, skp, skd);
				}
			}
			// if we want the spine to use the leg frames for PD movement as well, do this:
			for (int s = 0; s < controller->getLegFrameCount(); s++)
			{
				addJointToPDChain(controller->m_spine.getPDChain(), controller->getLegFrame(s)->m_legFrameJointId, skp, skd);
			}
			controller->m_spine.m_lfJointsUsedPD = true;
			// Fix the sub chains for our GCVF chain, count dof offsets
			int origGCDOFsz = controller->m_spine.m_DOFChainGravityCompForward.getSize();
			int oldJointGCIdx = -1;
			unsigned int vfIdx = 0;
			for (unsigned int m = 0; m < origGCDOFsz; m++)
			{
				unsigned int jointId = controller->m_spine.getGCVFChainFwd()->m_jointIdxChain[m];
				if (jointId != oldJointGCIdx)
				{
					controller->m_spine.getGCVFChainFwd()->m_jointIdxChainOffsets.push_back(m);
				}
				oldJointGCIdx = jointId;
			}
		}
		// FINALIZE
		// ------------------------------------------
		// Calculate number of torques axes in list, store
		unsigned int torqueListChunkSize = m_jointTorques.size() - torqueListOffset;
		controller->setTorqueListProperties(torqueListOffset, torqueListChunkSize);
		controller->handleInternalInitParamsConsume();
		// Add
		controller->setToBuildComplete();
		controller->m_sysIdx = m_controllers.size();
		m_controllers.push_back(controller);
		initControllerLocationAndVelocityStat((int)m_controllers.size() - 1, startGaitVelocity);
		// Finally, when all vars and lists have been built, add debug data
		// Add debug tracking for leg frame
#pragma region debugsetup
		if (i == 0)
		{
			dbgToolbar()->addReadOnlyVariable(Toolbar::CHARACTER, "Gait phase", Toolbar::FLOAT, (const void*)(controller->m_player.getPhasePointer()), " group='LegFrame'");
			// Velocity debug
			unsigned int vlistpos = m_controllerVelocityStats.size() - 1;
			dbgToolbar()->addReadOnlyVariable(Toolbar::CHARACTER, "Current velocity", Toolbar::DIR, (const void*)&m_controllerVelocityStats[vlistpos].m_currentVelocity, " group='LegFrame'");
			dbgToolbar()->addReadOnlyVariable(Toolbar::CHARACTER, "Desired velocity", Toolbar::DIR, (const void*)&m_controllerVelocityStats[vlistpos].m_desiredVelocity, " group='LegFrame'");
			dbgToolbar()->addReadWriteVariable(Toolbar::CHARACTER, "Goal velocity", Toolbar::DIR, (void*)&m_controllerVelocityStats[vlistpos].getGoalVelocity(), " group='LegFrame'");
			// Debug, per-lf stuff
			for (int n = 0; n < controller->getLegFrameCount(); n++) // leg frames
			{
				ControllerComponent::LegFrameEntityConstruct* legFrameEntities = controller->getLegFrameEntityConstruct(n);
				unsigned int legCount = legFrameEntities->m_upperLegEntities.size();
				ControllerComponent::LegFrame* legFrame = controller->getLegFrame(n);
				// Debug, per-leg stuff
				for (unsigned int x = 0; x < legCount; x++)
				{
					bool isLeft = x == 0;
					Color3f col = (isLeft ? Color3f(1.0f, 0.0f, 0.0f) : Color3f(0.0f, 1.0f, 0.0f));
					// Add debug tracking for leg
					std::string sideName = ToString(n)+(std::string(isLeft ? "Left" : "Right") + "Leg");
					dbgToolbar()->addReadWriteVariable(Toolbar::CHARACTER, (sideName.substr(0,2) + " Duty factor").c_str(), Toolbar::FLOAT, (void*)&legFrame->m_stepCycles[x].m_tuneDutyFactor, (" group='" + sideName + "'").c_str());
					dbgToolbar()->addReadWriteVariable(Toolbar::CHARACTER, (sideName.substr(0,2) + " Step trigger").c_str(), Toolbar::FLOAT, (void*)&legFrame->m_stepCycles[x].m_tuneStepTrigger, (" group='" + sideName + "'").c_str());
					// Foot strike placement visualization
					artemis::Entity & footPlcmtDbg = world->createEntity();
					footPlcmtDbg.addComponent(new RenderComponent());
					footPlcmtDbg.addComponent(new TransformComponent(glm::vec3(0.0f), glm::vec3(0.2f)));
					footPlcmtDbg.addComponent(new PositionRefComponent(&legFrame->m_footStrikePlacement[x]));
					footPlcmtDbg.addComponent(new MaterialComponent(col));
					footPlcmtDbg.refresh();
					// Foot lift placement visualization
					artemis::Entity & footLiftDbg = world->createEntity();
					footLiftDbg.addComponent(new RenderComponent());
					footLiftDbg.addComponent(new TransformComponent(glm::vec3(0.0f), glm::vec3(0.1f, 0.3f, 0.1f)));
					footLiftDbg.addComponent(new PositionRefComponent(&legFrame->m_footLiftPlacement[x]));
					footLiftDbg.addComponent(new MaterialComponent(col*0.7f));
					footLiftDbg.refresh();
					// Foot target placement visualization
					artemis::Entity & footTargetDbg = world->createEntity();
					footTargetDbg.addComponent(new RenderComponent());
					footTargetDbg.addComponent(new TransformComponent(glm::vec3(0.0f), glm::vec3(0.1f), glm::quat(glm::vec3((float)HALFPI, 0.0f, 0.0f))));
					footTargetDbg.addComponent(new PositionRefComponent(&legFrame->m_footTarget[x]));
					footTargetDbg.addComponent(new MaterialComponent(col*0.25f));
					footTargetDbg.refresh();
				}
			}
		}
#pragma endregion debugsetup
	}
	m_controllersToBuild.clear();
}

void ControllerSystem::addJointToVFChain(ControllerComponent::VFChain* p_VFChain, unsigned int p_idx, unsigned int p_vfIdx, const glm::vec3* p_angularLims /*= NULL*/)
{
	ControllerComponent::VFChain* legChain = p_VFChain;
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


void ControllerSystem::addJointToPDChain(ControllerComponent::PDChain* p_pdChain, unsigned int p_idx, float p_kp, float p_kd)
{
	ControllerComponent::PDChain* PDChain = p_pdChain;
	PDChain->m_PDChain.push_back(PDn(p_kp,p_kd));
	PDChain->m_jointIdxChain.push_back(p_idx);
}


void ControllerSystem::repeatAppendChainPart(ControllerComponent::VFChain* p_legVFChain, int p_localJointOffset, 
	int p_jointCount, unsigned int p_originalChainSize)
{
	ControllerComponent::VFChain* legChain = p_legVFChain;
	unsigned int DOFidx=0; // current dof being considerated for copy
	unsigned int totalJointsProcessed = 0;
	unsigned int jointAddedCounter = 0; // number of added joints
	unsigned int oldJointIdx = legChain->m_jointIdxChain[DOFidx];
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

unsigned int ControllerSystem::addJoint(RigidBodyComponent* p_jointRigidBody, TransformComponent* p_jointTransform, ControllerComponent* p_controllerParent)
{
	m_jointRigidBodies.push_back(p_jointRigidBody->getRigidBody());
	m_jointTorques.push_back(glm::vec3(0.0f));
	m_oldJointTorques.push_back(glm::vec3(0.0f));
	glm::mat4 matPosRot = p_jointTransform->getMatrixPosRot();
	m_jointWorldTransforms.push_back(matPosRot);
	m_jointLengths.push_back(p_jointTransform->getScale().y);
	m_jointMass.push_back(p_jointRigidBody->getMass());
	m_jointControllerParent.push_back(p_controllerParent);
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


// =================================================
// CONTROLLER UPDATE
// =================================================
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

	updateSpine(&m_jointTorques, p_controllerId, controller,p_dt);

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
	glm::vec3 goalV = m_controllerVelocityStats[p_controllerId].getGoalVelocity();
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
		if (goalSqrMag < currentSqrMag + stepSz*stepSz)
			desiredV = goalV;
		else if (currentV != glm::vec3(0.0f))
			desiredV += glm::normalize(currentV) * stepSz*p_dt;
	}
	else // if the goal is slower
	{
		// Take steps no smaller than 0.5
		if (goalSqrMag > currentSqrMag - stepSz*stepSz)
			desiredV = goalV;
		else if (currentV != glm::vec3(0.0f))
			desiredV -= glm::normalize(currentV) * stepSz*p_dt;
	}
	if (glm::length(desiredV) < 0.0f)
		DEBUGPRINT(( (string("dv: ")+ToString(desiredV.z)+"\n").c_str() ));
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

glm::vec3 ControllerSystem::getControllerStartPos(ControllerComponent* p_controller)
{
	return p_controller->getLegFrame(0)->m_startPosOffset;
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
		glm::vec3 goalV = m_controllerVelocityStats[p_controllerId].getGoalVelocity();
		glm::vec3 groundPos = m_controllerLocationStats[p_controllerId].m_currentGroundPos;
		float phi = p_controller->m_player.getPhase();
		for (unsigned int i = 0; i < legCount; i++)
		{
			//updateReferenceFootPosition(phi, m_runTime, goalV);
			updateFoot(p_controllerId, lf, i, phi, currentV, desiredV, groundPos);
		}
	}
}


void ControllerSystem::updateSpine(std::vector<glm::vec3>* p_outTVF, int p_controllerId, ControllerComponent* p_controller, float p_dt)
{
	ControllerComponent::Spine* spine = &p_controller->m_spine;
	if (spine->getPDChain()->getSize()>0 && p_controller->getLegFrameCount()>1)
	{
		float phi = p_controller->m_player.getPhase();
		glm::quat orientationDiff;
		// First, calculate the orientation diff between leg frames
		ControllerComponent::LegFrame* lf = p_controller->getLegFrame(0);
		glm::quat a = lf->getCurrentDesiredOrientation(phi);
		lf = p_controller->getLegFrame(1);
		glm::quat b = lf->getCurrentDesiredOrientation(phi);
		// The difference, ie. what will turn a into b
		orientationDiff = b*glm::inverse(a);

		// Then, apply this diff as the goal for the spines, divided by the amount of spine s
		ControllerComponent::PDChain* pdChain = spine->getPDChain();
		int spineJoints = pdChain->getSize();
		for (unsigned int x = 0; x < spineJoints; x++)
		{
			unsigned jointIdx = pdChain->m_jointIdxChain[x];
			// Calculate angle to leg frame space
			glm::quat current = glm::quat_cast(m_jointWorldTransforms[jointIdx]);
			// Drive PD using angle
			glm::vec3 torque = pdChain->m_PDChain[x].drive(current, orientationDiff / (float)spineJoints, p_dt);
			// Add to torque for joint
			(*p_outTVF)[jointIdx] += torque;
			glm::vec3 jointAxle = MathHelp::toVec3(m_jointWorldInnerEndpoints[jointIdx]);
			//dbgDrawer()->drawLine(jointAxle, jointAxle + torque*0.01f, dawnBringerPalRGB[x * 5], dawnBringerPalRGB[COL_LIGHTRED]);
		}
	}
	////////////////////
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
		//dbgDrawer()->drawLine(old, p_lf->m_footStrikePlacement[p_legIdx], dawnBringerPalRGB[COL_NAVALBLUE], dawnBringerPalRGB[COL_ORANGE]);
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
	glm::vec3 v = (p_velocity - p_desiredVelocity);
	v = glm::vec3(v.x, v.y, v.z);
	return p_footPosLF + v * p_lf->m_footPlacementVelocityScale;
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

	//// Compute the variants of torque and write to torque array
	//resetNonFeedbackJointTorques(&m_jointTorques, p_controller, p_controllerId, torqueIdxOffset, phi, p_dt);
	if (m_usePDTorque) computePDTorques(&m_jointTorques, p_controller, p_controllerId, torqueIdxOffset, phi, p_dt);
	computeAllVFTorques(&m_jointTorques, p_controller, p_controllerId, torqueIdxOffset, phi, p_dt);

	
	// Apply them to the leg frames, also
	// feed back corrections for hip joints
	if (m_useLFFeedbackTorque)
	{
		// Spine if it exists (calculated before leg frame loop)
		glm::vec3 tspine, tospine;
		ControllerComponent::Spine* spine = &p_controller->m_spine;
		// Note that the last two joints might be LFs, check the flag in spine to check for this wheter they should be used in this calculation or not
		int spineCount = (int)spine->m_joints;
		if (spineCount>0)
		{
			spineCount -= spine->m_lfJointsUsedPD ? 2 : 0; // dec with 2 if last two are LFs, assume for now we aren't counting those
			for (unsigned int i = 0; i < (unsigned int)spineCount; i++)
			{
				unsigned int spineIdx = spine->getPDChain()->m_jointIdxChain[i];
				tspine += m_jointTorques[spineIdx];
				tospine += m_oldJointTorques[spineIdx];
			}
		}

		glm::vec3 tLFremainder;
		// loop backwards, so the last remaining torque (ie. the front LF) returned is the one applied to the spine
		for (int i = ((int)p_controller->getLegFrameCount())-1; i>=0; i--)
		{
			tLFremainder=applyNetLegFrameTorque(p_controllerId, p_controller, i, tspine, tospine, phi, p_dt);
		}


		// If we're at the front LF, and we have a spine, add remainder work to the spine joints
		if (spineCount>0)
		{
			for (unsigned int i = 0; i < spineCount; i++)
			{
				unsigned int spineIdx = (unsigned int)spine->getPDChain()->m_jointIdxChain[i];
				m_jointTorques[spineIdx] += tLFremainder / (float)spineCount; // NOTE! Not sure if we should divide here? Or all joints assume the full torque?
			}
		}
	}
}

///-----------------------------------------------------------------------------------
/// Calculate the torque needed for the Leg Frame in order to keep it on its orientation
/// trajectory. This torque is feedbacked back through the stance legs, so this method
/// will correct those limbs' torques.
/// \param p_controllerIdx
/// \param p_lf
/// \param p_phi
/// \param p_dt
/// \param p_velocityStats
/// \return void
///-----------------------------------------------------------------------------------
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
			glm::vec3 fsw=calculateFsw(p_lf, i, p_phi, p_dt);
			glm::vec3 swingForce = calculateSwingLegVF(fsw);
			m_VFs[vfIdx] = swingForce;// Store force
			//
			//if (p_controllerIdx == 0)
			//	dbgDrawer()->drawLine(dbgFootPos, dbgFootPos + m_VFs[vfIdx], dawnBringerPalRGB[COL_PINK], dawnBringerPalRGB[COL_PURPLE]);
		}
		else
			// Stance force
		{
			glm::vec3 dbgFootPos = getFootPos(p_lf, i);
			//if (!stanceForcesCalculated)
			{
				fv = calculateFv(p_lf, m_controllerVelocityStats[p_controllerIdx]);
				//dbgDrawer()->drawLine(dbgFootPos, dbgFootPos + fv, dawnBringerPalRGB[COL_GREY]);
				fh = calculateFh(p_lf, m_controllerLocationStats[p_controllerIdx], p_phi, p_dt, glm::vec3(0.0f, 1.0f, 0.0));
				//dbgDrawer()->drawLine(dbgFootPos, dbgFootPos + fh, dawnBringerPalRGB[COL_BEIGE]);
				stanceForcesCalculated=true;
			}	
			glm::vec3 fd(calculateFd(p_controllerIdx,p_lf, i));
			glm::vec3 stanceForce = calculateStanceLegVF(stanceLegs, fv, fh, fd);
			m_VFs[vfIdx] = stanceForce;// Store force
			//if (p_controllerIdx == 0)
			//	dbgDrawer()->drawLine(dbgFootPos, dbgFootPos + m_VFs[vfIdx], dawnBringerPalRGB[COL_LIGHTBLUE], dawnBringerPalRGB[COL_NAVALBLUE]);
		}
	}

	delete[] legInStance;
}

///-----------------------------------------------------------------------------------
/// Method for calculating torque (and adding results to torque-array) from the 
/// previously defined Virtual Forces.
/// \param p_outTVF
/// \param p_controller
/// \param p_controllerIdx
/// \param p_torqueIdxOffset
/// \param p_phi
/// \param p_dt
/// \return void
///-----------------------------------------------------------------------------------
void ControllerSystem::computeAllVFTorques(std::vector<glm::vec3>* p_outTVF, ControllerComponent* p_controller, 
	unsigned int p_controllerIdx, unsigned int p_torqueIdxOffset, float p_phi, float p_dt)
{
	int spineCount = (int)p_controller->m_spine.m_joints;
	for (unsigned int i = 0; i < p_controller->getLegFrameCount(); i++)
	{
		ControllerComponent::VFChain* chain = NULL;
		ControllerComponent::LegFrame* lf = p_controller->getLegFrame(i);
		calculateLegFrameNetLegVF(p_controllerIdx, lf, p_phi, p_dt, m_controllerVelocityStats[p_controllerIdx]);
		// Begin calculating Jacobian transpose for each leg in leg frame
		unsigned int legCount = (unsigned int)lf->m_legs.size();	
		for (unsigned int n = 0; n < legCount; n++)
		{				
			ControllerComponent::Leg* leg = &lf->m_legs[n];
			if (m_useVFTorque)
			{
				chain = leg->getVFChain(ControllerComponent::STANDARD_CHAIN);
				computeVFTorquesFromChain(p_outTVF,chain, ControllerComponent::STANDARD_CHAIN, p_torqueIdxOffset, p_phi, p_dt);
			}
				
			if (m_useGCVFTorque && !isInControlledStance(lf, n, p_phi))
			{
				chain = leg->getVFChain(ControllerComponent::GRAVITY_COMPENSATION_CHAIN);
				computeVFTorquesFromChain(p_outTVF,chain, ControllerComponent::GRAVITY_COMPENSATION_CHAIN, p_torqueIdxOffset, p_phi, p_dt);
			}
		}	
		// also compute GCVF for spine joints
		// even though spine doesn't "belong" to a certain LF
		// we still have to compute the torque using both LFs as base
		// and weight based on distance
		if (m_useGCVFTorque && spineCount > 0)
		{
			chain = p_controller->m_spine.getGCVFChainFwd(); // front
			computeVFTorquesFromChain(p_outTVF,chain, ControllerComponent::GRAVITY_COMPENSATION_CHAIN, p_torqueIdxOffset, p_phi, p_dt);
			chain = p_controller->m_spine.getGCVFChainBwd(); // back
			computeVFTorquesFromChain(p_outTVF,chain, ControllerComponent::GRAVITY_COMPENSATION_CHAIN, p_torqueIdxOffset, p_phi, p_dt);
		}
	}
}

void ControllerSystem::computeVFTorquesFromChain(std::vector<glm::vec3>* p_outTVF, ControllerComponent::VFChain* p_vfChain,
	ControllerComponent::VFChainType p_type, unsigned int p_torqueIdxOffset, float p_phi, float p_dt)
{
	// Calculate torques using specified VF chain
	ControllerComponent::VFChain* chain = p_vfChain;
	unsigned int endJointIdx = chain->getEndJointIdx();
	unsigned int dofsToProcess = chain->getSize();
	unsigned int subChains = chain->m_jointIdxChainOffsets.size();
	for (int i = 0; i < max(1, subChains); i++) // always run at least once
	{		
		// get next end joint, if we're using subchains
		if (p_type == ControllerComponent::GRAVITY_COMPENSATION_CHAIN && subChains > 0)
		{
			unsigned int dofOffset = chain->m_jointIdxChainOffsets[i];
			endJointIdx = chain->m_jointIdxChain[dofOffset];
			if (i<subChains-1)
				dofsToProcess = chain->m_jointIdxChainOffsets[i+1];
			else
				dofsToProcess = chain->getSize();
		}
		// Get the end effector position
		// We're using the COM of the end joint in the chain
		// for a standard chain, this is equivalent to the foot
		glm::vec3 end = getJointPos(endJointIdx);
		//getFootPos(p_lf, p_legIdx);
		// Calculate the matrices
		CMatrix J = JacobianHelper::calculateVFChainJacobian(*chain,// Chain of DOFs to solve for
															end,							// Our end effector goal position
															&m_VFs,							// All virtual forces
															&m_jointWorldInnerEndpoints,	// All joint rotational axes
															&m_jointWorldTransforms,		// All joint world transformations
															dofsToProcess);					// Number of DOFs in list to work through
		CMatrix Jt = CMatrix::transpose(J);

		/*glm::mat4 sum(0.0f);
		for (unsigned int g = 0; g < m_jointWorldInnerEndpoints.size(); g++)
		{
		sum += m_jointWorldTransforms[g];
		}*/
		//DEBUGPRINT(((std::string("\n") + std::string(" WTransforms: ") + ToString(sum)).c_str()));
		//DEBUGPRINT(((std::string("\n") + std::string(" Pos: ") + ToString(end)).c_str()));
		//DEBUGPRINT(((std::string("\n") + std::string(" VF: ") + ToString(vf)).c_str()));

		// Use matrix to calculate and store torque
		for (unsigned int m = 0; m < dofsToProcess; m++)
		{
			// store torque
			unsigned int localJointIdx = chain->m_jointIdxChain[m];
			glm::vec3 vf = m_VFs[chain->m_vfIdxList[m]];

			// for visual force debug
			// ========================================================================
			if ((m_dbgShowGCVFVectors && p_type == ControllerComponent::VFChainType::GRAVITY_COMPENSATION_CHAIN)
				|| (m_dbgShowVFVectors && p_type == ControllerComponent::VFChainType::STANDARD_CHAIN))
			{
				glm::vec3 jpos = getJointPos(localJointIdx);
				Color3f lineCol = dawnBringerPalRGB[COL_LIMEGREEN];
				if (p_type == ControllerComponent::VFChainType::GRAVITY_COMPENSATION_CHAIN)
					lineCol = dawnBringerPalRGB[COL_PINK];
				//dbgDrawer()->drawLine(jpos, jpos + vf, lineCol);
			}
			// ========================================================================

			//glm::vec3 JjVec(J(0, m), J(1, m), J(2, m));
			glm::vec3 JVec(Jt(m, 0), Jt(m, 1), Jt(m, 2));
			glm::vec3 addT = (chain->m_DOFChain)[m] * glm::dot(JVec, vf);
			//
			//float ssum = JjVec.x + JjVec.y + JjVec.z;
			//DEBUGPRINT(((std::string("\n") + ToString(m) + std::string(" J sum: ") + ToString(ssum)).c_str()));
			//ssum = JVec.x + JVec.y + JVec.z;
			//DEBUGPRINT(((std::string("\n") + ToString(m) + std::string(" Jt sum: ") + ToString(ssum)).c_str()));
			/*bool vecnanchk = glm::isnan(addT) == glm::bool3(true, true, true);
			if (vecnanchk)
				int bb = 0;*/
			(*p_outTVF)[localJointIdx/* + p_torqueIdxOffset*/] += addT;
			// Do it like this for now, for the sake of readability and debugging.
		}

	} // next subchain(only used for GCVF chains for now)
}

bool ControllerSystem::isInControlledStance(ControllerComponent::LegFrame* p_lf, unsigned int p_legIdx, float p_phi)
{
	// Check if in stance and also read as stance if the 
	// foot is really touching the ground while in end of swing
	StepCycle* stepCycle = &p_lf->m_stepCycles[p_legIdx];
	bool stance = stepCycle->isInStance(p_phi);
	if (!stance)
	{
		bool isTouchingGround = false;
			//isFootStrike(p_lf,p_legIdx);
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
	glm::vec3 res;
	if (error > 0.0f)
	{
		glm::vec3 normdiff = glm::normalize(diff);
		float pdval = p_lf->m_footTrackingSpringDamper.drive(error, p_dt);
		glm::vec3 res = -normdiff * pdval;
		bool vecnanchk = glm::isnan(res) == glm::bool3(true, true, true);
		if (vecnanchk)
		{
			int i = 0;
		}
		glm::vec3 dbgFootPos = getFootPos(p_lf, p_legIdx);
		//dbgDrawer()->drawLine(dbgFootPos, dbgFootPos + res, dawnBringerPalRGB[COL_YELLOW]);
	}
	return res;
}




glm::vec3 ControllerSystem::calculateFv(ControllerComponent::LegFrame* p_lf, const VelocityStat& p_velocityStats)
{
	glm::vec3 fv=p_lf->m_velocityRegulatorKv*(p_velocityStats.m_desiredVelocity - p_velocityStats.m_currentVelocity);
	fv.y = 0.0f;
	return fv;
}

glm::vec3 ControllerSystem::calculateFh(ControllerComponent::LegFrame* p_lf, const LocationStat& p_locationStat, float p_phi, float p_dt, const glm::vec3& p_up)
{
	float hLF = p_lf->m_heightLFTraj.lerpGet(p_phi);
	glm::vec3 currentLocalHeight = p_locationStat.m_worldPos-p_locationStat.m_currentGroundPos;
	float currentHeightDeviation = currentLocalHeight.y-p_lf->m_height;
	// NOte that here we differ a little from the source material, as they use the diff between the wanted absolute
	// height and the current absolute height. Whereas we use the diff between the wanted deviation and the current deviation.
	// If they have a controller that falls below the wanted, they get hLF(phi)-h. If then h<hLF (below) they'll get positive force, and
	// if h>hLF they get negative. And that's what we need to keep for our solution, positive when below (to get up) and negative when
	// above (to get down). Note that this is a stance leg force, so thus the sign is ultimately switched when applied to the foot, in order
	// for the foot to push the body up (to get up) when h<hLF and thus the foot will itself get -fh.
	// the current height y only works for up=0,1,0
	// so in case we are making a space game, i'd reckon we should have the following PD work on vec3's
	// but for now, a float is OK
	float fh = p_lf->m_FhPD.drive(hLF - currentHeightDeviation, p_dt);// PD
	return p_up * fh;
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
	/*if (p_controllerId==0)
		dbgDrawer()->drawLine(getFootPos(p_lf, p_legIdx), getFootPos(p_lf, p_legIdx) + FD*2.0f, dawnBringerPalRGB[COL_YELLOW], dawnBringerPalRGB[COL_ORANGE]);*/
	//FD = glm::vec3(0.0f);
	return FD;
}

glm::vec3 ControllerSystem::calculateSwingLegVF(const glm::vec3& p_fsw)
{
	bool vecnanchk = glm::isnan(p_fsw) == glm::bool3(true, true, true);
	if (vecnanchk)
	{
		int i = 0;
	}
	return p_fsw; // Right now, this force is equivalent to fsw
}

glm::vec3 ControllerSystem::calculateStanceLegVF(unsigned int p_stanceLegCount,
	const glm::vec3& p_fv, const glm::vec3& p_fh, const glm::vec3& p_fd)
{
	float n = (float)p_stanceLegCount;
	glm::vec3 stanceVF = -p_fd - (p_fh / n) - (p_fv / n);
	bool vecnanchk = glm::isnan(stanceVF) == glm::bool3(true, true, true);
	if (vecnanchk)
	{
		int i = 0;
	}
	return stanceVF; // note fd should be stance fd
}


glm::vec3 ControllerSystem::applyNetLegFrameTorque(unsigned int p_controllerId, ControllerComponent* p_controller, unsigned int p_legFrameIdx, 
	glm::vec3& p_tspine, glm::vec3& p_tospine, float p_phi, float p_dt)
{
	// Preparations, get a hold of all legs in stance,
	// all legs in swing. And get a hold of their and the 
	// closest spine's torques.
	ControllerComponent::LegFrame* lf = p_controller->getLegFrame(p_legFrameIdx);
	unsigned int legCount = (unsigned int)lf->m_legs.size();	
	unsigned int lfJointIdx = lf->m_legFrameJointId;
	glm::vec3 tstance(0.0f), tswing(0.0f), tspine(0.0f), // current frame
		tostance(0.0f), toswing(0.0f), tospine(0.0f); // previous frame
	unsigned int stanceCount = 0;
	unsigned int* stanceLegBuf = new unsigned int[legCount];
	for (unsigned int i = 0; i < legCount; i++)
	{
		unsigned int jointId = lf->m_hipJointId[i];
		glm::vec3 jTorque=m_jointTorques[jointId], 
			joTorque = m_oldJointTorques[jointId];
		if (isInControlledStance(lf, i, p_phi))
		{
			tstance += jTorque;
			tostance += joTorque;
			stanceLegBuf[stanceCount] = jointId;
			stanceCount++;
		}
		else
		{
			tswing += jTorque;
			toswing += joTorque;
		}
	}
	tspine = p_tspine;
	tospine = p_tospine;


	// 1. Calculate current torque for leg frame:
	// tLF = tstance + tswing + tspine.
	// Here the desired torque is feedbacked through the
	// stance legs (see 3) as their current torque
	// is the product of previous desired torque combined
	// with current real-world scenarios.
	glm::vec3 tLF = tostance + toswing + tospine; // ie. est. what we got now, base don last torque action
	m_jointTorques[lfJointIdx] = tLF;
	/*if (p_controllerId==0)
		dbgDrawer()->drawLine(getLegFramePosition(lf), getLegFramePosition(lf) + tLF, dawnBringerPalRGB[COL_PURPLE], dawnBringerPalRGB[COL_YELLOW]);*/

	// 2. Calculate a desired torque, tdLF, using the previous current
	// torque, tLF, and a PD-controller driving towards the 
	// desired orientation, omegaLF.
	glm::mat4 dWM = getDesiredWorldOrientation(p_controllerId);
	//glm::vec3 hgr = MathHelp::transformDirection(dWM, glm::vec3(5.0f, 0.0f, 0.0f));
	//glm::vec3 fram = MathHelp::transformDirection(dWM, glm::vec3(0.0f, 0.0f, 5.0f));
	//glm::mat4 up = glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0), glm::vec3(0.0f, 1.0f, 0.0));
	glm::quat desiredW = glm::quat_cast(dWM/**up*/);
	glm::quat omegaLF = lf->getCurrentDesiredOrientation(p_phi);
	omegaLF = /*desiredW**/omegaLF;
	glm::quat currentOrientation = MathHelp::getMatrixRotation(m_jointWorldTransforms[lf->m_legFrameJointId]);
	glm::mat4 orient = glm::mat4_cast(currentOrientation);
	glm::vec3 tdLF = lf->getOrientationPDTorque(currentOrientation, omegaLF, p_dt);

	//glm::vec3 chgr = MathHelp::transformDirection(orient, glm::vec3(5.0f, 0.0f, 0.0f));
	//glm::vec3 cfram = MathHelp::transformDirection(orient, glm::vec3(0.0f, 0.0f, 5.0f));

	// Draw ORIENTATION distance
	//glm::vec3 wanted = MathHelp::transformDirection(glm::mat4_cast(omegaLF), glm::vec3(0.0f, 10.0f, 0.0f));
	//glm::vec3 current = MathHelp::transformDirection(glm::mat4_cast(currentOrientation), glm::vec3(0.0f, 10.0f, 0.0f));
	/*if (p_controllerId == 0)
	{
		// torque
		dbgDrawer()->drawLine(getLegFramePosition(lf), getLegFramePosition(lf) + tdLF, dawnBringerPalRGB[COL_ORANGE], dawnBringerPalRGB[COL_YELLOW]);

	}*/
	// test code
	//rigidbody.AddTorque(tdLF);

	// How much torque should be assumed by the LF?
	// The remainder will be assumed by the spine.
	bool spineWork = p_legFrameIdx == 0 /*&& (p_controller->getLegFrameCount()>1)*/;
	float percentage = 1.0f;
	if (spineWork) percentage = 0.5f;
	// For the front LF we want only 50% (as per the document)


	// 3. Now loop through all legs in stance (N) and
	// modify their torques in the vector according
	// to tstancei = (tdLF-tswing-tspine)/N
	// This is to try to make the stance legs compensate
	// for current errors in order to make the leg frame
	// reach its desired torque.

	int N = stanceCount;
	glm::vec3 ntLF = tdLF - tswing - tspine; // the new TLF we're striving for
	glm::vec3 work = percentage*((ntLF) / (float)N);
	// We multiply by 0.5f here if we're at the front LF (see above) as the quadruped model will assume
	// 50% of the required torque and the stance legs will take the remaining
	// torque. I'm currently trying this for both the quadruped and biped models.
	// Apply to stance legs
	for (int i = 0; i < N; i++)
	{
		unsigned int idx = stanceLegBuf[i];
		// here we use the wanted tLF and subtract the current swing and spine torques
		m_jointTorques[idx] = work;
	}

	delete[] stanceLegBuf;
	// The vector reference to the torques, now contains the new LF torque
	// as well as any corrected stance-leg torques.
	return (1.0f - percentage)*ntLF; // return the remaining new TLF we want to achieve
									 // by returning we can let an optional spine be adjusted
									 // to supply the remainder
}



///-----------------------------------------------------------------------------------
/// Calculate (and accumulate to torque-array) the torque from each joint's PD-driver. This is the torque that follows
/// the IK animation movement (the IK drives the PD's). This torque is meant to retain
/// the internal "shape" of a leg (ie. bent, or straight).
/// \param p_outTVF
/// \param p_controller
/// \param p_controllerIdx
/// \param p_torqueIdxOffset
/// \param p_phi
/// \param p_dt
/// \return void
///-----------------------------------------------------------------------------------
void ControllerSystem::computePDTorques(std::vector<glm::vec3>* p_outTVF, ControllerComponent* p_controller, 
	unsigned int p_controllerIdx, unsigned int p_torqueIdxOffset, float p_phi, float p_dt)
{
	glm::mat4 desiredOrientation = getDesiredWorldOrientation(p_controllerIdx);
	glm::mat4 invDesiredOrientation = glm::inverse(desiredOrientation);
	int lfCount = p_controller->getLegFrameCount();
	for (unsigned int i = 0; i < lfCount; i++)
	{
		unsigned int lfIdx = i;
		ControllerComponent::LegFrame* lf = p_controller->getLegFrame(lfIdx);
		glm::quat currentOrientationQuat = MathHelp::getMatrixRotation(getLegFrameTransform(lf));
		glm::mat4 currentOrientation = glm::mat4_cast(currentOrientationQuat);
		unsigned int legCount = (unsigned int)lf->m_legs.size();
		LocationStat* locationStat = &m_controllerLocationStats[p_controllerIdx];
		// for each leg
		for (unsigned int n = 0; n < legCount; n++)
		{
			/*
				The plane in which the IK chain
				acts is defined by a normal that is fixed in the leg - frame coordinate
				system, and the location of the relevant shoulder or hip joint.
				*/
			IK2Handler* ik = &lf->m_legIK[n];
			ControllerComponent::Leg* leg = &lf->m_legs[n];
			ControllerComponent::PDChain* pdChain = leg->getPDChain();
			// Fetch foot and hip reference pos
			glm::vec3 refDesiredFootPos = lf->m_footTarget[n];
			//refDesiredFootPos.y -= m_jointLengths[pdChain->getFootJointIdx()] * 0.5f;
			float dist = 1.0f;
			if (ik->getKneeFlip() < 0) dist = -2.0f;
			refDesiredFootPos.y -= lf->m_footHeight*0.5f*ik->getKneeFlip();
			refDesiredFootPos.z -= dist*m_jointLengths[pdChain->getFootJointIdx()] * 0.5f;
			//refDesiredFootPos.z -= 0.2f;
			glm::vec3 refHipPos = MathHelp::toVec3(m_jointWorldInnerEndpoints[lf->m_hipJointId[n]]); // TODO TRANSFORM FROM WORLD SPACE TO LOCAL AND THEN BACK AGAIN FOR PD
			/*if (lfCount<=1) */refHipPos.y = locationStat->m_currentGroundPos.y + lf->m_height - m_jointLengths[lf->m_legFrameJointId] * 0.5f;
			// Fetch upper- and lower leg length and solve IK
			DebugDrawBatch* drawer = NULL;
			if (p_controllerIdx == 0) drawer = dbgDrawer();
			ik->solve(refDesiredFootPos, refHipPos,
				1.05f*m_jointLengths[pdChain->getUpperJointIdx()],
				1.05f*m_jointLengths[pdChain->getLowerJointIdx()] /*+ lf->m_footHeight*/, drawer);
			// For each PD in legb
			for (unsigned int x = 0; x < pdChain->getSize(); x++)
			{
				float sagittalAngle = 0.0f;
				// Fetch correct angle based on segment type
				// remember that the legs points "downwards when angle=0"
				// 90 forward = -HALFPI (counterclockwise)
				// 90 deg backward = HALFPI (clockwise)
				// But the ik has other angle space 
				glm::quat referenceFrame = currentOrientationQuat;
				if (x == pdChain->getUpperLegSegmentIdx())
				{
					sagittalAngle = -(ik->getUpperLegAngle() + PI*0.5f);
					referenceFrame = glm::quat_cast(desiredOrientation);
				}
				else if (x == pdChain->getLowerLegSegmentIdx())
					sagittalAngle = -(ik->getLowerWorldLegAngle() + PI*0.5f);
				else if (x == pdChain->getFootIdx())
				{
					sagittalAngle = getDesiredFootAngle(n, lf, p_phi);
					referenceFrame = glm::quat_cast(desiredOrientation);
				}
				//
				unsigned jointIdx = pdChain->m_jointIdxChain[x];
				// Calculate angle to leg frame space
				glm::quat goal = /*desiredOrientationY * */glm::quat(glm::vec3(sagittalAngle, 0.0f, 0.0f));
				glm::quat current = glm::quat_cast(m_jointWorldTransforms[jointIdx]);
				// Drive PD using angle
				glm::vec3 torque = pdChain->m_PDChain[x].drive(current, goal, p_dt);
				//bool vecnanchk = glm::isnan(torque) == glm::bool3(true, true, true);
				// Add to torque for joint
				(*p_outTVF)[jointIdx] += torque;
				/*glm::vec3 jointAxle = MathHelp::toVec3(m_jointWorldInnerEndpoints[jointIdx]);
				if (p_controllerIdx == 0)
					dbgDrawer()->drawLine(jointAxle, jointAxle + torque*0.01f, dawnBringerPalRGB[x*5], dawnBringerPalRGB[COL_CORNFLOWERBLUE]);*/
			}
		}
	}
	//throw std::exception("The method or operation is not implemented.");
	// For each leg:
	//  Fetch IK
	//  Fetch foot reference pos
	//  Fetch upper- and lower leg length
	//  Solve IK
	//	For each PD in leg:
	//	  If less than foot id:
	//	    Fetch  correct solved angle from IK
	//	    Drive PD with angle
	//	    Fetch result from PD, add to torque
	//    else if is foot id:
	//		applyFootTorque

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
	if (dir.x == 0.0f && dir.z==0.0f) dir += glm::vec3(0.0f, 0.0f, 0.0001f);
	dir.z *= -1.0f; // as we're using a ogl function below, we flip z
	return glm::lookAt(glm::vec3(0.0f), glm::normalize(dir), glm::vec3(0.0f, 1.0f, 0.0f));
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
		for (unsigned int x = 0; x < legCount; x++)
		{
			unsigned int footRBIdx = lf->m_footRigidBodyIdx[x];
			lf->m_footIsColliding[x] = m_rigidBodyRefs[footRBIdx]->isColliding();
		}
	}
	// phew
}

float ControllerSystem::getDesiredFootAngle(unsigned int p_legIdx, ControllerComponent::LegFrame* p_lf, float p_phi)
{
	// Toe-off is modeled with a linear target trajectory towards a fixed toe-off
	// target angle Theta-A that is triggered DeltaT-A seconds in advance of the start of the swing phase,
	// as dictated by the gait graph.
	//
	// Foot- strike anticipation is done in an analogous fashion with
	// respect to the anticipated foot-strike time and defined by Theta-B and DeltaT-B.
	//
	StepCycle* stepcycle = &p_lf->m_stepCycles[p_legIdx];
	//Transform foot = m_feet[i].transform;
	float toeOffStartOffset = p_lf->m_toeOffTime[p_legIdx];
	float footStrikeStartOffset = p_lf->m_tuneFootStrikeTime[p_legIdx];
	// get absolutes
	// Get point in time when toe lifts off from ground
	float toeOffStart = stepcycle->m_tuneStepTrigger + stepcycle->m_tuneDutyFactor - toeOffStartOffset;
	if (toeOffStart >= 1.0f) toeOffStart -= 1.0f;
	if (toeOffStart < 0.0f) toeOffStart += 1.0f;
	// Get point in time when foot touches the ground
	float footStrikeStart = stepcycle->m_tuneStepTrigger - footStrikeStartOffset;
	if (toeOffStart < 0.0f) footStrikeStart += 1.0f;
	//         

	// return correct angle for PD
	float result = 0.0f;
	if ((p_phi > footStrikeStart && footStrikeStart > toeOffStart) || 
		p_phi < toeOffStart) // catch first
		result = p_lf->m_tuneFootStrikeAngle;
	else
	{
		result = p_lf->m_tuneToeOffAngle;
	}
	return result;
}

ControllerSystem::VelocityStat& ControllerSystem::getControllerVelocityStat(const ControllerComponent* p_controller)
{
	return m_controllerVelocityStats[p_controller->m_sysIdx];
}

glm::vec3 ControllerSystem::getJointAcceleration(unsigned int p_jointId)
{
	unsigned int idx = p_jointId;
	glm::vec3 result;
	if (idx < m_jointRigidBodies.size())
	{
		result = m_rigidBodyRefs[idx]->getAcceleration();
	}
	return result;
}

double ControllerSystem::getLatestTiming()
{
	return m_timing;
}

glm::vec3 ControllerSystem::getJointPos(unsigned int p_jointIdx)
{
	return MathHelp::getMatrixTranslation(m_jointWorldTransforms[p_jointIdx]);
}

glm::vec3 ControllerSystem::getJointOuterPos(unsigned int p_jointIdx)
{
	return MathHelp::toVec3(m_jointWorldOuterEndpoints[p_jointIdx]);
}

glm::vec3 ControllerSystem::getJointInnerPos(unsigned int p_jointIdx)
{
	return MathHelp::toVec3(m_jointWorldInnerEndpoints[p_jointIdx]);
}

