#include "ControllerComponent.h"
#include <OptimizableHelper.h>

ControllerComponent::ControllerComponent(artemis::Entity* p_legFrame, 
	std::vector<artemis::Entity*>& p_hipJoints)
{
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
	m_legFrames.push_back(legFrame);
}

std::vector<float> ControllerComponent::getParams()
{
	std::vector<float> params;
	params.push_back(m_player.getParams());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		params.push_back(m_legFrames[i].getParams());
	}
}


void ControllerComponent::consumeParams(const std::vector<float>& p_other)
{
	m_player.consumeParams(p_other);
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		m_legFrames[i].consumeParams();
	}
}

std::vector<float> ControllerComponent::getParamsMax()
{
	std::vector<float> paramsmax;
	paramsmax.push_back(m_player.getParamsMax());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		paramsmax.push_back(m_legFrames[i].getParamsMax());
	}
}

std::vector<float> ControllerComponent::getParamsMin()
{
	std::vector<float> paramsmin;
	paramsmin.push_back(m_player.getParamsMin());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		paramsmin.push_back(m_legFrames[i].getParamsMin());
	}
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
	std::vector<float> params;
	// All per leg frame data
	for (int i = 0; i < 3; i++)
		params.push_back(m_orientationLFTraj[i].getParams());	//
	params.push_back(m_heightLFTraj.getParams());			//
	params.push_back(m_footTrackingGainKp.getParams());	//
	params.push_back(m_stepHeighTraj.getParams());			//
	params.push_back(m_footTransitionEase.getParams());	//
	/*params.push_back(m_desiredLFTorquePD;		//*/
	params.push_back(m_FhPD.getParams());					// optimiziable height force pd
	params.push_back(m_lateStrikeOffsetDeltaH);
	params.push_back(m_velocityRegulatorKv);
	params.push_back(OptimizableHelper::ExtractParamsListFrom(m_FDHVComponents));
	params.push_back(m_footPlacementVelocityScale);			// per leg frame
	/*float			 m_height;*/
	params.push_back(OptimizableHelper::ExtractParamsListFrom(m_stepLength));
	params.push_back(m_tuneToeOffAngle);
	params.push_back(m_tuneFootStrikeAngle);
	// All per leg data
	for (int i = 0; i < m_legs.size(); i++)
	{
		params.push_back(m_stepCycles[i].getParams());
		params.push_back(m_toeOffTime[i]);
		params.push_back(m_tuneFootStrikeTime[i]);
	}


}

void ControllerComponent::LegFrame::consumeParams(const std::vector<float>& p_other)
{

}

std::vector<float> ControllerComponent::LegFrame::getParamsMax()
{

}

std::vector<float> ControllerComponent::LegFrame::getParamsMin()
{

}
