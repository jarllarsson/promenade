#include "ControllerOptimizationSystem.h"
#include "ControllerMovementRecorderComponent.h"

int ControllerOptimizationSystem::m_testCount = 0;

ControllerOptimizationSystem::ControllerOptimizationSystem()
{
	addComponentType<ControllerComponent>();
	addComponentType<ControllerMovementRecorderComponent>();
	// settings
	m_simTicks = 1.0f;			
	m_warmupTicks = 0.0f;	
	m_instantEval = false;
	// playback
	m_currentSimTicks = 0.0f;	
	m_currentBestCandidateIdx = -1;
	m_lastBestScore = FLT_MAX;
	m_firstControllerAdded = false;
	//
	m_controllerSystemRef = NULL;
};

void ControllerOptimizationSystem::added(artemis::Entity &e)
{
	ControllerComponent* controller = controllerComponentMapper.get(e);
	ControllerMovementRecorderComponent* recorder = controllerRecorderComponentMapper.get(e);
	if (!m_firstControllerAdded)
	{
		m_paramsMax = controller->getParamsMax();
		m_paramsMin = controller->getParamsMin();
	}
	m_optimizableControllers.push_back(controller);
	m_controllerRecorders.push_back(recorder);
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
	bool foundBetter = false;
	for (int i = 0; i < m_controllerScores.size(); i++)
	{
		if (m_controllerScores[i] < bestScore)
		{
			m_currentBestCandidateIdx = i;
			bestScore = m_controllerScores[i];
			foundBetter = true;
		}
	}
	m_lastBestScore = bestScore;
	if (foundBetter) m_lastBestParams = m_currentParams[m_currentBestCandidateIdx];
	/*if (m_currentBestCandidateIdx > -1)
		m_drawBestCandidate = m_currentBestCandidate;*/
}

void ControllerOptimizationSystem::voidBestCandidate()
{
	m_currentBestCandidateIdx = -1;
}

void ControllerOptimizationSystem::storeParams( std::vector<float>* p_initParams/*=NULL*/ )
{
	m_currentParams.clear();
	if (p_initParams==NULL)
	{
		for (int i = 0; i < m_optimizableControllers.size(); i++)
			m_currentParams.push_back(m_optimizableControllers[i]->getParams());
	}
	else
	{
		for (int i = 0; i < m_optimizableControllers.size(); i++)
			m_currentParams.push_back(*p_initParams);
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
	// Perturb and assign to candidates
	for (int i = p_offset; i < m_optimizableControllers.size(); i++)
	{
		m_currentParams[i] = m_changer.change(m_lastBestParams, m_paramsMin, m_paramsMax, m_testCount); // different perturbation to each
	}
}

void ControllerOptimizationSystem::evaluateAll()
{
	for (int i = 0; i < m_controllerRecorders.size(); i++)
	{
		//Debug.Log("Eval "+i+" "+m_optimizableControllers[i]);
		m_controllerScores[i] += evaluateCandidateFitness(i);
	}
}

double ControllerOptimizationSystem::evaluateCandidateFitness(int p_idx)
{
	ControllerMovementRecorderComponent* record = m_controllerRecorders[p_idx];
	double score = record->evaluate();
	return score;
}


void ControllerOptimizationSystem::initSim( std::vector<float>* p_initParams/*=NULL*/ )
{
	unsigned int sz = m_optimizableControllers.size();
	m_controllerScores=std::vector<double>(sz); // All scores for one round
	m_currentParams = std::vector<std::vector<float>>(sz);
	//
	storeParams(p_initParams);
	m_lastBestParams = m_currentParams[0];
	perturbParams(1);

	for (int i = 0; i < m_optimizableControllers.size(); i++)
	{
		IOptimizable* opt = static_cast<IOptimizable*>(m_optimizableControllers[i]);
		std::vector<float> paramslist = m_currentParams[i];
		opt->consumeParams(paramslist); // consume it to controller
	}
	resetScores();
}

void ControllerOptimizationSystem::processEntity(artemis::Entity &e)
{
	ControllerComponent* controller = controllerComponentMapper.get(e);
	ControllerMovementRecorderComponent* recorder = controllerRecorderComponentMapper.get(e);

	// record:
	recorder->fv_calcStrideMeanVelocity(controller, m_controllerSystemRef);
	recorder->fr_calcRotationDeviations(controller, m_controllerSystemRef);
	recorder->fh_calcHeadAccelerations(controller);
	recorder->fd_calcReferenceMotion(controller);
	recorder->fp_calcMovementDistance(controller, m_controllerSystemRef);
}

// Call after eval
std::vector<float>& ControllerOptimizationSystem::getWinnerParams()
{
	return m_lastBestParams;
}
