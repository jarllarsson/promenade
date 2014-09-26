#include "ControllerComponent.h"
#include <OptimizableHelper.h>
#include <DebugPrint.h>

ControllerComponent::ControllerComponent(artemis::Entity* p_legFrame, 
	std::vector<artemis::Entity*>& p_hipJoints)
{
	m_player = GaitPlayer(2.0f);
	//
	m_buildComplete = false;
	// for each inputted leg-frame entity...

	// Set up the entity-based leg frame representation
	// This is simply a struct of pointers to the artemis equivalents of
	// what the controller system will work with as joints and decomposed DOF-chains
	LegFrameEntityConstruct legFrameEntityConstruct;
	legFrameEntityConstruct.m_legFrameEntity = p_legFrame;
	// Add all legs to it
	for (unsigned int i = 0; i < p_hipJoints.size(); i++)
	{
		legFrameEntityConstruct.m_upperLegEntities.push_back(p_hipJoints[i]);
	}
	// add to our list of constructs
	m_legFrameEntityConstructs.push_back(legFrameEntityConstruct);
	// Create the leg frame data struct as well
	// Allocate it according to number of leg entities that was inputted
	LegFrame legFrame;
	legFrame.m_stepCycles.resize(legFrameEntityConstruct.m_upperLegEntities.size());
	legFrame.m_stepCycles[1].m_tuneStepTrigger = 0.5f;
	//legFrame.m_stepCycles[0].m_tuneDutyFactor=1.0f;
	//legFrame.m_stepCycles[1].m_tuneDutyFactor=1.0f;
	m_legFrames.push_back(legFrame);
}

ControllerComponent::ControllerComponent(std::vector<artemis::Entity*>& p_legFrames, 
	std::vector<artemis::Entity*>& p_hipJoints)
{
	m_player = GaitPlayer(2.0f);
	//
	m_buildComplete = false;
	// for each inputted leg-frame entity...

	// Set up the entity-based leg frame representation
	// This is simply a struct of pointers to the artemis equivalents of
	// what the controller system will work with as joints and decomposed DOF-chains
	int hipJointsIdx = 0;
	for (int i = 0; i < p_legFrames.size(); i++)
	{
		LegFrameEntityConstruct legFrameEntityConstruct;
		legFrameEntityConstruct.m_legFrameEntity = p_legFrames[i];
		// Add all legs to it
		for (int n=0; n<2; n++)
		{
			legFrameEntityConstruct.m_upperLegEntities.push_back(p_hipJoints[hipJointsIdx]);
			hipJointsIdx++;
		}
		// add to our list of constructs
		m_legFrameEntityConstructs.push_back(legFrameEntityConstruct);
		// Create the leg frame data struct as well
		// Allocate it according to number of leg entities that was inputted
		LegFrame legFrame;
		legFrame.m_stepCycles.resize(legFrameEntityConstruct.m_upperLegEntities.size());
		legFrame.m_stepCycles[0].m_tuneStepTrigger = (i==0?0.0f:0.5f); // flip offset, on step trigger based on whether
		legFrame.m_stepCycles[1].m_tuneStepTrigger = (i==0?0.5f:0.0f); // it is a front- or back LF
		//legFrame.m_stepCycles[0].m_tuneDutyFactor=1.0f;
		//legFrame.m_stepCycles[1].m_tuneDutyFactor=1.0f;
		m_legFrames.push_back(legFrame);
	}
}

std::vector<float> ControllerComponent::getParams()
{
	//DEBUGPRINT(("\nCONTROLLER COMP GETPARAMS\n"));
	std::vector<float> params;
	OptimizableHelper::addRange(params,m_player.getParams());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		OptimizableHelper::addRange(params,m_legFrames[i].getParams());
	}
	return params;
}


void ControllerComponent::consumeParams(std::vector<float>& p_other)
{
	m_player.consumeParams(p_other);
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		m_legFrames[i].consumeParams(p_other);
	}
}

std::vector<float> ControllerComponent::getParamsMax()
{
	std::vector<float> paramsmax;
	OptimizableHelper::addRange(paramsmax, m_player.getParamsMax());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		OptimizableHelper::addRange(paramsmax, m_legFrames[i].getParamsMax());
	}
	return paramsmax;
}

std::vector<float> ControllerComponent::getParamsMin()
{
	std::vector<float> paramsmin;
	OptimizableHelper::addRange(paramsmin, m_player.getParamsMin());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		OptimizableHelper::addRange(paramsmin, m_legFrames[i].getParamsMin());
	}
	return paramsmin;
}

unsigned int ControllerComponent::getHeadJointId()
{
	return m_legFrames[0].m_legFrameJointId; // currently no head exist, so check the frame itself
}

////////////////////////////////////////////////////////////////////////////
// LEG-FRAME

glm::quat ControllerComponent::LegFrame::getCurrentDesiredOrientation(float p_phi)
{
	float yaw = m_orientationLFTraj[(unsigned int)Orientation::YAW].lerpGet(p_phi);
	float pitch = m_orientationLFTraj[(unsigned int)Orientation::PITCH].lerpGet(p_phi);
	float roll = m_orientationLFTraj[(unsigned int)Orientation::ROLL].lerpGet(p_phi);
	return glm::quat(glm::vec3(pitch, yaw, roll)); // radians
}

glm::vec3 ControllerComponent::LegFrame::getOrientationPDTorque(const glm::quat& p_currentOrientation, const glm::quat& p_desiredOrientation, float p_dt)
{
	glm::vec3 torque = m_desiredLFTorquePD.drive(p_currentOrientation, p_desiredOrientation, p_dt);
	return torque;
}

void ControllerComponent::LegFrame::createFootPlacementModelVarsForNewLeg(const glm::vec3& p_startPos)
{
	m_footStrikePlacement.push_back(p_startPos);
	m_footLiftPlacement.push_back(p_startPos);
	m_footLiftPlacementPerformed.push_back(false);
	m_footTarget.push_back(p_startPos);
	m_footIsColliding.push_back(false);
}

std::vector<float> ControllerComponent::LegFrame::getParams()
{
	//DEBUGPRINT(("LEG FRAME GETPARAMS\n"));
	std::vector<float> params;
	// All per leg frame data
	for (int i = 0; i < 3; i++)
		OptimizableHelper::addRange(params,m_orientationLFTraj[i].getParams());	//
	OptimizableHelper::addRange(params,m_heightLFTraj.getParams());			//
	OptimizableHelper::addRange(params,m_footTrackingGainKp.getParams());	//
	OptimizableHelper::addRange(params,m_stepHeighTraj.getParams());			//
	OptimizableHelper::addRange(params,m_footTransitionEase.getParams());	//
	/*OptimizableHelper::addRange(params,m_desiredLFTorquePD;		//*/
	OptimizableHelper::addRange(params,m_FhPD.getParams());					// optimizable height force pd
	/*OptimizableHelper::addRange(params,m_lateStrikeOffsetDeltaH);*/
	params.push_back(m_velocityRegulatorKv);
	OptimizableHelper::addRange(params,OptimizableHelper::ExtractParamsListFrom(m_FDHVComponents));
	params.push_back(m_footPlacementVelocityScale);			// per leg frame
	/*float			 m_height;*/
	OptimizableHelper::addRange(params,OptimizableHelper::ExtractParamsListFrom(m_stepLength));
	params.push_back(m_tuneToeOffAngle);
	params.push_back(m_tuneFootStrikeAngle);
	// All per leg data
	for (int i = 0; i < m_legs.size(); i++)
	{
		OptimizableHelper::addRange(params,m_stepCycles[i].getParams());
		params.push_back(m_toeOffTime[i]);
		params.push_back(m_tuneFootStrikeTime[i]);
	}
	return params;
}

void ControllerComponent::LegFrame::consumeParams(std::vector<float>& p_other)
{
	for (int i = 0; i < 3; i++)
		m_orientationLFTraj[i].consumeParams(p_other);	//
	m_heightLFTraj.consumeParams(p_other);			//
	m_footTrackingGainKp.consumeParams(p_other);	//
	m_stepHeighTraj.consumeParams(p_other);			//
	m_footTransitionEase.consumeParams(p_other);	//
	m_FhPD.consumeParams(p_other);					// optimizable height force pd
	OptimizableHelper::ConsumeParamsTo(p_other, &m_velocityRegulatorKv);
	OptimizableHelper::ConsumeParamsTo(p_other, &m_FDHVComponents);
	OptimizableHelper::ConsumeParamsTo(p_other, &m_footPlacementVelocityScale);	
	OptimizableHelper::ConsumeParamsTo(p_other, &m_stepLength);
	OptimizableHelper::ConsumeParamsTo(p_other, &m_tuneToeOffAngle);
	OptimizableHelper::ConsumeParamsTo(p_other, &m_tuneFootStrikeAngle);
	// All per leg data
	for (int i = 0; i < m_legs.size(); i++)
	{
		m_stepCycles[i].consumeParams(p_other);
		OptimizableHelper::ConsumeParamsTo(p_other,&m_toeOffTime[i]);
		OptimizableHelper::ConsumeParamsTo(p_other,&m_tuneFootStrikeTime[i]);
	}
}

std::vector<float> ControllerComponent::LegFrame::getParamsMax()
{
	std::vector<float> paramsmax;
	// All per leg frame data
	for (int i = 0; i < 3; i++)
	for (int x = 0; x < m_orientationLFTraj[i].getSize(); x++)
		paramsmax.push_back(TWOPI);		// m_orientationLFTraj
	for (int i = 0; i < m_heightLFTraj.getSize(); i++)			paramsmax.push_back(1.0f);		// heightLFTraj
	for (int i = 0; i < m_footTrackingGainKp.getSize(); i++)	paramsmax.push_back(1.0f);		// footTrackingGainKp
	for (int i = 0; i < m_stepHeighTraj.getSize(); i++)			paramsmax.push_back(1.5f);		// stepHeighTraj
	for (int i = 0; i < m_footTransitionEase.getSize(); i++)	paramsmax.push_back(1.0f);		// footTransitionEase
	/*OptimizableHelper::addRange(params,m_desiredLFTorquePD;		//*/
	paramsmax.push_back(100.0f); paramsmax.push_back(10.0f); // FhPD (kp, kd), optimizable height force pd
	/*paramsmax.push_back(m_lateStrikeOffsetDeltaH);*/
	paramsmax.push_back(3.0f); // velocityRegulatorKv
	for (int i=0;i<4;i++) paramsmax.push_back(2.0f);	// FDHVComponents;
	paramsmax.push_back(2.0f);	// footPlacementVelocityScale
	paramsmax.push_back(0.2f); paramsmax.push_back(3.3f);// step length
	paramsmax.push_back(TWOPI); // toe off angle
	paramsmax.push_back(TWOPI); // foot strike angle
	// All per leg data
	for (int i = 0; i < m_legs.size(); i++)
	{
		OptimizableHelper::addRange(paramsmax, m_stepCycles[i].getParamsMax());
		paramsmax.push_back(0.5f); // toe off time
		paramsmax.push_back(0.5f); // foot strike time
	}
	return paramsmax;
}

std::vector<float> ControllerComponent::LegFrame::getParamsMin()
{
	std::vector<float> paramsmin;
	// All per leg frame data
	for (int i = 0; i < 3; i++)
		for (int x = 0; x < m_orientationLFTraj[i].getSize(); x++)
			paramsmin.push_back(0.0f);		// m_orientationLFTraj
	for (int i = 0; i < m_heightLFTraj.getSize(); i++)			paramsmin.push_back(0.0f);		// heightLFTraj
	for (int i = 0; i < m_footTrackingGainKp.getSize(); i++)	paramsmin.push_back(0.0f);		// footTrackingGainKp
	for (int i = 0; i < m_stepHeighTraj.getSize(); i++)			paramsmin.push_back(0.0f);		// stepHeighTraj
	for (int i = 0; i < m_footTransitionEase.getSize(); i++)	paramsmin.push_back(0.0f);		// footTransitionEase
	/*OptimizableHelper::addRange(params,m_desiredLFTorquePD;		//*/
	paramsmin.push_back(0.0f); paramsmin.push_back(0.0f); // FhPD (kp, kd), optimizable height force pd
	/*paramsmax.push_back(m_lateStrikeOffsetDeltaH);*/
	paramsmin.push_back(0.0f); // velocityRegulatorKv
	for (int i = 0; i<4; i++) paramsmin.push_back(-2.0f);	// FDHVComponents;
	paramsmin.push_back(0.0f);	// footPlacementVelocityScale
	paramsmin.push_back(0.0f); paramsmin.push_back(0.0f);// step length
	paramsmin.push_back(0.0f); // toe off angle
	paramsmin.push_back(0.0f); // foot strike angle
	// All per leg data
	for (int i = 0; i < m_legs.size(); i++)
	{
		OptimizableHelper::addRange(paramsmin, m_stepCycles[i].getParamsMin());
		paramsmin.push_back(0.0f); // toe off time
		paramsmin.push_back(0.0f); // foot strike time
	}
	return paramsmin;
}
