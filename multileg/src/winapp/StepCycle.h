#pragma once

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


class StepCycle
{
public:
	// Fraction of overall normalized time for which the 
	// foot is touching the ground.
	float m_tuneDutyFactor;

	// Offset time point in normalized time when the
	// foot begins its cycle.
	float m_tuneStepTrigger;

	StepCycle();


	// IOptimizable
	/*
	vector<float> GetParams()
	{
		List<float> vals = new List<float>();
		vals.Add(m_tuneDutyFactor);
		vals.Add(m_tuneStepTrigger);
		return vals;
	}

	public void ConsumeParams(List<float> p_params)
	{
		m_tuneDutyFactor = p_params[0];
		m_tuneStepTrigger = p_params[1];
		for (int i = 0; i < 2; i++)
			p_params.RemoveAt(0);
		sanitize();
	}

	public List<float> GetParamsMax()
	{
		List<float> maxList = new List<float>();
		maxList.Add(0.99f);
		maxList.Add(0.99f);
		return maxList;
	}

	public List<float> GetParamsMin()
	{
		List<float> minList = new List<float>();
		minList.Add(0.001f);
		minList.Add(0.001f);
		return minList;
	}
	*/



	bool isInStance(float p_phi);


	//
	// Get phase of swing, is zero in stance (0-1-0)
	// can be used as period in transition function
	//
	float getSwingPhase(float p_phi);

	float getStancePhase(float p_phi);

private:
	void sanitize();
};