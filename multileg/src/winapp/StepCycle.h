#pragma once
#include <IOptimizable.h>

// =======================================================================================
//                                      StepCycle
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Class containing data pertaining to one step cycle. Also has accessibility
///         methods for automatically wrap-access data.
///			NOTE! Might be in need of optimization
///        
/// # StepCycle
/// 
/// 3-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------


class StepCycle : public IOptimizable
{
public:
	// Fraction of overall normalized time for which the 
	// foot is touching the ground.
	float m_tuneDutyFactor;

	// Offset time point in normalized time when the
	// foot begins its cycle.
	float m_tuneStepTrigger;

	StepCycle();


	bool isInStance(float p_phi);


	//
	// Get phase of swing, is zero in stance (0-1-0)
	// can be used as period in transition function
	//
	float getSwingPhase(float p_phi);

	float getStancePhase(float p_phi);

	// Optimization
	virtual std::vector<float> getParams();
	virtual void consumeParams(std::vector<float>& p_other);
	virtual std::vector<float> getParamsMax();
	virtual std::vector<float> getParamsMin();

private:
	void sanitize();
};