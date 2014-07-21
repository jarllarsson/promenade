#pragma once
#include <IOptimizable.h>
#include <OptimizableHelper.h>
#include <DebugPrint.h>

// =======================================================================================
//                                      GaitPlayer
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Class that acts as an "animation player". It keeps track of the timeline in
///			regards of the gait cycle speed.
///        
/// # GaitPlayer
/// 
/// 3-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class GaitPlayer : public IOptimizable
{
public:
	GaitPlayer()
	{
		m_gaitPhase = 0.0f;
		m_tuneGaitPeriod = 1.0f;
	}



	// IOptimizable
	/*public List<float> GetParams()
	{
		List<float> vals = new List<float>();
		vals.Add(m_tuneGaitPeriod);
		return vals;
	}

	public void ConsumeParams(List<float> p_params)
	{
		OptimizableHelper.ConsumeParamsTo(p_params, ref m_tuneGaitPeriod);
	}

	public List<float> GetParamsMax()
	{
		List<float> maxList = new List<float>();
		maxList.Add(3.0f);
		return maxList;
	}

	public List<float> GetParamsMin()
	{
		List<float> minList = new List<float>();
		minList.Add(0.01f);
		return minList;
	}*/

	void updatePhase(float p_t)
	{
		m_gaitPhase += p_t / m_tuneGaitPeriod;
		while (m_gaitPhase > 1.0f)
		{
			m_gaitPhase -= 1.0f;
			m_hasRestarted_oneCheck = true;
		}
	}

	bool checkHasRestartedStride_AndResetFlag() // ugh...
	{
		bool res = m_hasRestarted_oneCheck;
		m_hasRestarted_oneCheck = false;
		return res;
	}

	float getPhase()
	{
		return m_gaitPhase;
	}

	const float* getPhasePointer()
	{
		return &m_gaitPhase;
	}

	// Optimization
	virtual std::vector<float> getParams()
	{
		DEBUGPRINT(("GAIT PLAYER GETPARAMS\n"));
		std::vector<float> params;
		params.push_back(m_tuneGaitPeriod);
		return params;
	}
	virtual void consumeParams(std::vector<float>& p_other)
	{
		OptimizableHelper::ConsumeParamsTo(p_other, &m_tuneGaitPeriod);
	}
	virtual std::vector<float> getParamsMax()
	{
		std::vector<float> paramsmax;
		paramsmax.push_back(5.0f);
		return paramsmax;
	}
	virtual std::vector<float> getParamsMin()
	{
		std::vector<float> paramsmin;
		paramsmin.push_back(0.01f);
		return paramsmin;
	}

private:
	// Current gait phase
	float m_gaitPhase; // phi

	// Total gait time (ie. stride duration)
	float m_tuneGaitPeriod; // T


	//
	bool m_hasRestarted_oneCheck; // can be read once after each restart of stride
};