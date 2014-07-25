#include "ControllerMovementRecorderComponent.h"
#include "ControllerComponent.h"
#include "ControllerSystem.h"


ControllerMovementRecorderComponent::ControllerMovementRecorderComponent()
{
	m_fdWeight = 100.0f;
	m_fvWeight = 5.0f;
	m_fhWeight = 0.5f;
	m_frWeight = 5.0f;
	m_fpWeight = 0.5f;
}


double ControllerMovementRecorderComponent::evaluate()
{
	double fv = evaluateFV();
	double fr = evaluateFR();
	double fh = evaluateFH();
	double fp = evaluateFP();
	double fd = evaluateFD();
	double fobj = (double)m_fdWeight*fd + (double)m_fvWeight*fv + (double)m_frWeight*fr + (double)m_fhWeight*fh - (double)m_fpWeight*fp;
	//Debug.Log(fobj+" = "+(double)m_fvWeight*fv+" + "+(double)m_frWeight*fr+" + "+(double)m_fhWeight*fh+" - "+(double)m_fpWeight*fp);
	return fobj;
}

void ControllerMovementRecorderComponent::fv_calcStrideMeanVelocity(ControllerComponent* p_controller,
	ControllerSystem* p_system, bool p_forceStore /*= false*/)
{
	GaitPlayer* player = &p_controller->m_player;
	bool restarted = player->checkHasRestartedStride_AndResetFlag();
	ControllerSystem::VelocityStat& velocities = p_system->getControllerVelocityStat(p_controller);
	if (!restarted && !p_forceStore)
	{
		m_temp_currentStrideVelocities.push_back(velocities.m_currentVelocity);
		m_temp_currentStrideDesiredVelocities.push_back(velocities.m_desiredVelocity);
	}
	else
	{
		glm::vec3 totalVelocities(0.0f), totalDesiredVelocities(0.0f);
		for (int i = 0; i < m_temp_currentStrideVelocities.size(); i++)
		{
			totalVelocities += m_temp_currentStrideVelocities[i];
			// force straight movement behavior from tests, set desired coronal velocity to constant zero:
			totalDesiredVelocities += glm::vec3(0.0f, 0.0f, m_temp_currentStrideDesiredVelocities[i].z);
		}
		totalVelocities /= (float)m_temp_currentStrideVelocities.size();
		totalDesiredVelocities /= (float)m_temp_currentStrideDesiredVelocities.size();
		// add to lists
		m_fvVelocityDeviations.push_back((double)glm::length(totalVelocities - totalDesiredVelocities));
		//
		m_temp_currentStrideVelocities.clear();
		m_temp_currentStrideDesiredVelocities.clear();
	}
}

void ControllerMovementRecorderComponent::fr_calcRotationDeviations(ControllerComponent* p_controller, ControllerSystem* p_system)
{
	unsigned int legframes = p_controller->getLegFrameCount();
	GaitPlayer* player = &p_controller->m_player;
	for (int i = 0; i < legframes; i++)
	{
		ControllerComponent::LegFrame* lf = p_controller->getLegFrame(i);
		glm::quat currentDesiredOrientation = lf->getCurrentDesiredOrientation(player->getPhase());
		glm::quat currentOrientation = MathHelp::getMatrixRotation(p_system->getLegFrameTransform(lf));
		glm::quat diff = glm::inverse(currentOrientation) * currentDesiredOrientation;
		glm::vec3 axis; float angle;
		MathHelp::quatToAngleAxis(diff, angle, axis);
		m_frBodyRotationDeviations[i].push_back(angle);
	}
}

void ControllerMovementRecorderComponent::fh_calcHeadAccelerations( ControllerComponent* p_controller )
{
	//m_fhHeadAcceleration.Add((double)m_myController.m_headAcceleration.magnitude);
}

void ControllerMovementRecorderComponent::fd_calcReferenceMotion( ControllerComponent* p_controller )
{
	/*
	double lenBod = (double)m_myController.transform.position.y - (double)m_origBodyHeight;
	lenBod *= lenBod; // sqr
	double lenHd = (double)m_myController.m_head.transform.position.y - (double)m_origHeadHeight;
	lenHd *= lenHd; // sqr
	double ghostDist = (double)(m_ghostController.position.z - m_ghostStart.z);
	double controllerDist = (double)(m_myController.transform.position.z - m_mycontrollerStart.z);
	double lenDist = ghostDist - controllerDist;
	if (controllerDist < 0.0) lenDist *= 2.0; // penalty for falling or walking backwards
	lenDist *= lenDist; // sqr

	double lenFt = 0.0;
	double lenHips = 0.0;
	double lenKnees = 0.0;
	Vector3 wantedWPos = new Vector3(m_myController.transform.position.x, m_origBodyHeight, m_ghostController.position.z - m_ghostStart.z + m_mycontrollerStart.z);
	wantedWPos = new Vector3(m_myController.transform.position.x, m_origBodyHeight, m_myController.transform.position.z);
	for (int i = 0; i < m_myLegFrame.m_feet.Length; i++)
	{
		Vector3 footRefToFoot = (m_myLegFrame.m_feet[i].transform.position - wantedWPos) - (m_referenceHandler.m_foot[i].position - m_ghostController.position);
		Vector3 hipRefToHip = (m_myController.m_joints[(i * 2) + 1].transform.position - wantedWPos) - (m_referenceHandler.m_IK[i].m_hipPos - m_ghostController.position);
		Vector3 kneeRefToKnee = (m_myController.m_joints[(i + 1) * 2].transform.position - wantedWPos) - (m_referenceHandler.m_knee[i].position - m_ghostController.position);
		Debug.DrawLine(m_myLegFrame.m_feet[i].transform.position,
			m_myLegFrame.m_feet[i].transform.position - footRefToFoot, Color.white);
		Debug.DrawLine(m_myController.m_joints[(i * 2) + 1].transform.position,
			m_myController.m_joints[(i * 2) + 1].transform.position - hipRefToHip, Color.yellow*2.0f);
		Debug.DrawLine(m_myController.m_joints[(i + 1) * 2].transform.position,
			m_myController.m_joints[(i + 1) * 2].transform.position - kneeRefToKnee, Color.yellow);

		lenFt += (double)Vector3.SqrMagnitude(footRefToFoot);
		lenHips += (double)Vector3.SqrMagnitude(hipRefToHip);
		lenKnees += (double)Vector3.SqrMagnitude(kneeRefToKnee);
	}
	m_fdBodyHeightSqrDiffs.Add(lenFt * 0.4 + lenKnees + lenHips + lenBod + 2.0f*lenHd + 0.1 * lenDist);*/
}

void ControllerMovementRecorderComponent::fp_calcMovementDistance(ControllerComponent* p_controller, ControllerSystem* p_system)
{
	m_fpMovementDist.push_back(p_system->getControllerVelocityStat(p_controller).m_currentVelocity);
	m_fvVelocityGoal = p_system->getControllerVelocityStat(p_controller).m_goalVelocity;
}

double ControllerMovementRecorderComponent::evaluateFV()
{
	double total = 0.0f;
	// mean
	for (auto n : m_fvVelocityDeviations)
	{
		total += n;
	}
	double avg = total /= max(1.0, ((double)(m_fvVelocityDeviations.size())));
	double totmeandiffsqr = 0.0f;
	// std
	for (auto n : m_fvVelocityDeviations)
	{
		double mdiff = n - avg;
		totmeandiffsqr += mdiff * mdiff;
	}
	double sdeviation = sqrt(totmeandiffsqr / max(1.0, ((double)m_fvVelocityDeviations.size())));
	return avg;
}

double ControllerMovementRecorderComponent::evaluateFR()
{
	double total = 0.0f;
	int sz = 0;
	// mean
	for (int x = 0; x < m_frBodyRotationDeviations.size();x++)
	for (int y = 0; y < m_frBodyRotationDeviations[x].size(); y++)
	{
		total += (double)m_frBodyRotationDeviations[x][y];
		sz++;
	}
	double avg = total /= max(1, (double)(sz));
	return avg;
}

double ControllerMovementRecorderComponent::evaluateFH()
{
	double total = 0.0f;
	int sz = 0;
	// mean
	for(auto n : m_fhHeadAcceleration)
	{
		total += n;
		sz++;
	}

	double avg = total /= max(1.0, (double)(sz));
	return avg;
}

double ControllerMovementRecorderComponent::evaluateFD()
{
	double total = 0.0f;
	// mean
	for (auto n : m_fdBodyHeightSqrDiffs)
	{
		total += n;
	}
	double avg = total /= max(1.0, (double)(m_fdBodyHeightSqrDiffs.size()));
	double totmeandiffsqr = 0.0f;
	// std
	for (auto n : m_fdBodyHeightSqrDiffs)
	{
		double mdiff = n - avg;
		totmeandiffsqr += mdiff * mdiff;
	}
	double sdeviation = sqrt(totmeandiffsqr / max(1.0, (double)m_fdBodyHeightSqrDiffs.size()));
	return avg;
}

double ControllerMovementRecorderComponent::evaluateFP()
{
	glm::vec3 total;
	// mean
	for (auto n : m_fpMovementDist)
	{
		total += n;
	}
	float movementSign = max(0.1f, m_fvVelocityGoal.z) / abs(max(0.1f, m_fvVelocityGoal.z));
	if (movementSign == 0.0) movementSign = 1.0f;
	double scoreInRightDir = max(0.0f, total.z * movementSign);
	return scoreInRightDir;
}
