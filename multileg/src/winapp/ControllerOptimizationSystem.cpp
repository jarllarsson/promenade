#include "ControllerOptimizationSystem.h"

int ControllerOptimizationSystem::m_testCount = 0;

ControllerOptimizationSystem::ControllerOptimizationSystem()
{
	addComponentType<ControllerComponent>();
	// settings
	m_simTicks = 1.0f;			
	m_warmupTicks = 0.0f;	
	m_instantEval = false;
	// playback
	m_currentSimTicks = 0.0f;	
	m_currentBestCandidateIdx = -1;
	m_lastBestScore = FLT_MAX;
	m_firstControllerAdded = false;
};

void ControllerOptimizationSystem::added(artemis::Entity &e)
{
	ControllerComponent* controller = controllerComponentMapper.get(e);
	if (!m_firstControllerAdded)
	{
		m_paramsMax = controller->getParamsMax();
		m_paramsMin = controller->getParamsMin();
	}
	m_firstControllerAdded = true;
}

void ControllerOptimizationSystem::resetTestCount()
{
	m_testCount = 0;
}

void ControllerOptimizationSystem::incTestCount()
{
	m_testCount++;
}

float ControllerOptimizationSystem::getCurrentSimTime()
{
	return m_currentSimTicks;
}

bool ControllerOptimizationSystem::isSimCompleted()
{
	return m_currentSimTicks >= m_simTicks;
}

void ControllerOptimizationSystem::restartSim()
{

}

void ControllerOptimizationSystem::findCurrentBestCandidate()
{

}

void ControllerOptimizationSystem::voidBestCandidate()
{

}

void ControllerOptimizationSystem::storeParams()
{

}

void ControllerOptimizationSystem::resetScores()
{

}

void ControllerOptimizationSystem::perturbParams(int p_offset /*= 0*/)
{

}

void ControllerOptimizationSystem::evaluateAll()
{

}

double ControllerOptimizationSystem::evaluateCandidateFitness(int p_idx)
{

}
