#pragma once
#include "IK2Handler.h"
#include "StepCycle.h"
#include "GaitPlayer.h"
#include "ControllerComponent.h"
#include "PieceWiseLinear.h"

// =======================================================================================
//                            ReferenceLegMovementController
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # ReferenceLegMovementController
/// Detailed description.....
/// Created on: 5-8-2014 
///---------------------------------------------------------------------------------------

class ReferenceLegMovementController
{
public:
	ReferenceLegMovementController(ControllerComponent* p_controller, ControllerComponent::LegFrame* p_lf, 
		unsigned int p_legCount=2)
	{
		unsigned int legs = p_legCount;
		for (int i = 0; i < legs; i++)
		{
			m_feet.push_back(glm::vec3());
			m_knees.push_back(glm::vec3());
			m_oldFeetPos.push_back(glm::vec3());
			m_liftPos.push_back(glm::vec3());
			m_stepCycles.push_back(p_lf->m_stepCycles[i]);
			m_stepLength = p_lf->m_stepLength;
		}
		m_stepHeightTraj = p_lf->m_stepHeighTraj;
		m_player = p_controller->m_player;
	}

	ReferenceLegMovementController(const ReferenceLegMovementController& p_copy)
	{
		m_feet			= p_copy.m_feet;
		m_knees			= p_copy.m_knees;
		m_oldFeetPos	= p_copy.m_oldFeetPos;
		m_stepCycles	= p_copy.m_stepCycles;
		m_player		= p_copy.m_player;
		m_stepLength		= p_copy.m_stepLength;
		m_stepHeightTraj	= p_copy.m_stepHeightTraj;
		m_liftPos			= p_copy.m_liftPos;
	}

	virtual ~ReferenceLegMovementController() {}

	IK2Handler				m_IK;
	std::vector<glm::vec3>	m_feet;
	std::vector<glm::vec3>	m_knees;
	std::vector<glm::vec3>	m_oldFeetPos;
	std::vector<StepCycle>	m_stepCycles;
	GaitPlayer				m_player;

	// PLF, the coronal(x) and sagittal(y) step distance
	glm::vec2 m_stepLength;

	PieceWiseLinear m_stepHeightTraj;
	std::vector<glm::vec3> m_liftPos;

	void updateRefPositions(unsigned int p_legIdx, const glm::vec3& p_lfPos, float p_lfHeight, float p_uLegLen, float p_lLegLen, float p_dt, DebugDrawBatch* p_drawer)
	{
		// Advance the player
		m_player.updatePhase(p_dt);
		float phi = m_player.getPhase();
		//
		
		bool inStance = m_stepCycles[p_legIdx].isInStance(phi);
		//				
		float flip = (p_legIdx * 2.0f) - 1.0f;
		glm::vec3 baseOffset(flip*m_stepLength.x, 0.0f, 0.0f);
		glm::vec3 lfPlaneBase(p_lfPos.x, 0.0f, p_lfPos.z);
		glm::vec3 lfHeightBase(0.0f, p_uLegLen+p_lLegLen, 0.0f);
		if (!inStance)
		{
			float swingPhi = m_stepCycles[p_legIdx].getSwingPhase(phi);
			// The height offset, ie. the "lift" that the foot makes between stepping points.
			glm::vec3 heightOffset(0.0f, m_stepHeightTraj.lerpGet(swingPhi), 0.0f);

			glm::vec3 wpos = glm::lerp(m_liftPos[p_legIdx],
				p_lfPos + glm::vec3(baseOffset.x, 0.0f, m_stepLength.y*0.5f),
				swingPhi);
			wpos = glm::vec3(wpos.x, 0.0f, wpos.z);
			m_feet[p_legIdx] = wpos + heightOffset;

		}
		else
		{
			m_liftPos[p_legIdx] = m_feet[p_legIdx];
			//Debug.DrawLine(m_foot[i].position, m_foot[i].position + Vector3.up, Color.magenta - new Color(0.3f, 0.3f, 0.3f, 0.0f), 10.0f);
		}
		//Color debugColor = Color.red;
		//if (i == 1) debugColor = Color.green;
		//Debug.DrawLine(m_oldFootPos[i], m_foot[i].position, debugColor, 30.0f);
		m_oldFeetPos[p_legIdx] = m_feet[p_legIdx];
		m_IK.solve(m_feet[p_legIdx], baseOffset + lfPlaneBase + lfHeightBase, p_uLegLen, p_lLegLen, p_drawer);
		//
		m_knees[p_legIdx] = m_IK.getKneePos()/* + lfPlaneBase (already world)*/;
	}


};