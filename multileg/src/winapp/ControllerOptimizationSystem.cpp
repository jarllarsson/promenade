#include "ControllerOptimizationSystem.h"

ControllerOptimizationSystem::ControllerOptimizationSystem()
{
	addComponentType<ControllerComponent>();
	// settings
	m_simTime = 1.0f;			
	m_warmupTime = 0.0f;	
	m_instantEval = false;
	// playback
	m_currentSimTime = 0.0f;	
	m_currentBestCandidateIdx = -1;
	m_lastBestScore = FLT_MAX;
};
