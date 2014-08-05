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
	ReferenceLegMovementController(ControllerComponent* p_controller, ControllerComponent::LegFrame* p_lf)
	{
		unsigned int legs = p_lf->m_legs.size();
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
	virtual ~ReferenceLegMovementController();

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
	//public PcswiseLinear m_tuneFootTransitionEase;

// 	void Awake()
// 	{
// 		for (int i = 0; i < m_feet.size(); i++)
// 		{
// 			m_oldFeetPos[i] = m_feet[i];
// 		}
// 	}

	// Use this for initialization
//	void Start()
//	{
// 		for (int i = 0; i < m_foot.Length; i++)
// 		{
// 			m_liftPos[i] = m_foot[i].position;
// 		}

	// Update is called once per frame
	void iterate(float p_dt)
	{
		// Advance the player
		m_player.updatePhase(p_dt);

		updateFeetPositions(m_player.getPhase());
		updateKneeRefs();
	}

	void updateKneeRefs()
	{
		for (int i = 0; i < m_knees.size(); i++)
		{
			m_knees[i] = m_IK[i].m_kneePosW;
		}
	}


	void updateFeetPositions(float p_phi)
	{
		for (int i = 0; i < m_foot.Length; i++)
		{
			bool inStance = m_stepCycles[i].isInStance(p_phi);
			//
			if (!inStance)
			{
				float swingPhi = m_stepCycles[i].getSwingPhase(p_phi);
				// The height offset, ie. the "lift" that the foot makes between stepping points.
				Vector3 heightOffset = new Vector3(0.0f, m_stepHeightTraj.getValAt(swingPhi), 0.0f);
				float flip = (i * 2.0f) - 1.0f;
				Vector3 wpos = Vector3.Lerp(m_liftPos[i],
					transform.position + new Vector3(flip*m_stepLength.x, 0.0f, m_stepLength.y*0.5f),
					swingPhi);
				wpos = new Vector3(wpos.x, 0.0f, wpos.z);
				m_foot[i].position = wpos + heightOffset;

			}
			else
			{
				m_liftPos[i] = m_foot[i].position;
				Debug.DrawLine(m_foot[i].position, m_foot[i].position + Vector3.up, Color.magenta - new Color(0.3f, 0.3f, 0.3f, 0.0f), 10.0f);
			}
			Color debugColor = Color.red;
			if (i == 1) debugColor = Color.green;
			Debug.DrawLine(m_oldFootPos[i], m_foot[i].position, debugColor, 30.0f);
			m_oldFootPos[i] = m_foot[i].position;
		}
	}

};