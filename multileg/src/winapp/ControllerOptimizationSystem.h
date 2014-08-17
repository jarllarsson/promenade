#pragma once
#include "AdvancedEntitySystem.h"
#include "ControllerComponent.h"
#include "ControllerMovementRecorderComponent.h"
#include <ParamChanger.h>
#include "ControllerSystem.h"

// =======================================================================================
//                                      ControllerOptimizationSystem
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Measures the controllers and perform parameter evaluation on simulation end.
///			New parameters are provided as well when restarting.
///			Note that fixed frame step size is expected for running optimization sim, 
///			for fully deterministic eval.
///        
/// # ControllerOptimizationSystem
/// 
/// 16-7-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ControllerOptimizationSystem : public AdvancedEntitySystem
{
private:
	artemis::ComponentMapper<ControllerComponent> controllerComponentMapper;
	artemis::ComponentMapper<ControllerMovementRecorderComponent> controllerRecorderComponentMapper;

	ControllerComponent* m_bestScoreController;
	ParamChanger m_changer;
	// Note that fixed frame step size is expected for running optimization sim, for fully deterministic eval
	int m_simTicks; // The amount of ticks the sim will be run
	int m_warmupTicks; // The amount of ticks to ignore taking measurements for score
	int m_currentSimTicks; // The current playback ticks
	bool m_instantEval; // whether to evaluate the score at once (1 sim tick)

	int m_currentBestCandidateIdx;

	double m_lastBestScore; // "Hiscore" (the lower, the better)
	std::vector<double> m_controllerScores; // All scores for one round
	std::vector<float> m_lastBestParams; // saved params needed for a controller to get the hiscore
	std::vector<float> m_paramsMax; // Prefetch of controller parameter
	std::vector<float> m_paramsMin; // bounds in the current sim
	std::vector<std::vector<float> > m_currentParams; // all params for the current controllers
	std::vector<ControllerComponent*> m_optimizableControllers;
	std::vector<ControllerMovementRecorderComponent*> m_controllerRecorders;

	static int m_testCount; // global amount of executed tests

	ControllerSystem* m_controllerSystemRef;
public:

	ControllerOptimizationSystem();

	virtual void initialize()
	{
		controllerComponentMapper.init(*world);
		controllerRecorderComponentMapper.init(*world);
		m_controllerSystemRef = (ControllerSystem*)(world->getSystemManager()->getSystem<ControllerSystem>());
	};

	virtual void removed(artemis::Entity &e)
	{

	}

	virtual void added(artemis::Entity &e);

	virtual void processEntity(artemis::Entity &e);

	virtual void fixedUpdate(float p_dt)
	{
		m_time += p_dt;

	}

	void initSim(double p_hiscore, std::vector<float>* p_initParams=NULL);
	static void resetTestCount();
	int getCurrentSimTicks();
	void incSimTick();
	void stepTime(double p_dt);
	bool isSimCompleted();	
	void evaluateAll();
	std::vector<float>& getWinnerParams();
	double getWinnerScore();
	void findCurrentBestCandidate();
	double getScoreOf(unsigned int p_idx);
	std::vector<float>* getCurrentParamsOf(unsigned int p_idx);
	std::vector<float> getParamsOf(unsigned int p_idx);
protected:
private:
	static void incTestCount();
	void populateControllerInitParams();

	void restartSim();
	void voidBestCandidate();
	void storeParams(std::vector<float>* p_initParams=NULL);
	void resetScores();
	void perturbParams(int p_offset = 0);

	double evaluateCandidateFitness(int p_idx);

	//bool m_firstControllerAdded;
	bool m_inited;
	double m_time;
};
