#include "ControllerOptimizationSystem.h"
#include "ControllerMovementRecorder.h"

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
	m_optimizableControllers.push_back(controller);
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
	m_currentSimTicks = -m_warmupTicks;
	voidBestCandidate();
	resetScores();
}

void ControllerOptimizationSystem::findCurrentBestCandidate()
{
	voidBestCandidate();
	double bestScore = m_lastBestScore;
	for (int i = 0; i < m_controllerScores.size(); i++)
	{
		if (m_controllerScores[i] < bestScore)
		{
			m_currentBestCandidateIdx = i;
			bestScore = m_controllerScores[i];
		}
	}
	m_lastBestScore = bestScore;
	/*if (m_currentBestCandidateIdx > -1)
		m_drawBestCandidate = m_currentBestCandidate;*/
}

void ControllerOptimizationSystem::voidBestCandidate()
{
	m_currentBestCandidateIdx = -1;
}

void ControllerOptimizationSystem::storeParams()
{
	m_currentParams.clear();
	for (int i = 0; i < m_optimizableControllers.size(); i++)
	{
		m_currentParams.push_back(m_optimizableControllers[i]->getParams());
	}
}

void ControllerOptimizationSystem::resetScores()
{
	for (int i = 0; i < m_controllerScores.size(); i++)
	{
		m_controllerScores[i] = 0.0f;
	}
}

void ControllerOptimizationSystem::perturbParams(int p_offset /*= 0*/)
{
	// Get params from winner and use as basis for perturbing
	// Only update params if they were better than before, else reuse old
	if (m_currentBestCandidateIdx > -1)
	{
		//IOptimizable best = m_optimizableControllers[m_currentBestCandidate];
		m_lastBestParams = m_currentParams[m_currentBestCandidateIdx];
	}
	// Perturb and assign to candidates
	for (int i = p_offset; i < m_optimizableControllers.size(); i++)
	{
		m_currentParams[i] = m_changer.change(m_lastBestParams, m_paramsMin, m_paramsMax, m_testCount); // different perturbation to each
	}
}

void ControllerOptimizationSystem::evaluateAll()
{
	for (int i = 0; i < m_optimizableControllers.size(); i++)
	{
		//Debug.Log("Eval "+i+" "+m_optimizableControllers[i]);
		m_controllerScores[i] += evaluateCandidateFitness(i);
	}
}

/////////////////////////////////////////////////////////
//// TO DO!!!!!!!!!!
/////////////////////////////////////////////////////////
double ControllerOptimizationSystem::evaluateCandidateFitness(int p_idx)
{
	ControllerMovementRecorder* record = m_optimizableControllers[p_idx]->getRecordedData();
	double score = record->evaluate();
	return score;
}


void ControllerOptimizationSystem::initSim()
{
	unsigned int sz = m_optimizableControllers.size();
	m_controllerScores=std::vector<double>(sz); // All scores for one round
	m_currentParams = std::vector<std::vector<float>>(sz);
	//
	storeParams();
	m_lastBestParams = m_currentParams[0];
	perturbParams(2);
	for (int i = 2; i < m_optimizableControllers.size(); i++)
	{
		IOptimizable* opt = static_cast<IOptimizable*>(m_optimizableControllers[i]);
		std::vector<float> paramslist = m_currentParams[i];
		opt->consumeParams(paramslist); // consume it to controller
	}
	resetScores();
}
