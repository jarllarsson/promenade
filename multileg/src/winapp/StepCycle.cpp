#include "StepCycle.h"
#include <OptimizableHelper.h>
#include <DebugPrint.h>

StepCycle::StepCycle()
{
	m_tuneDutyFactor = 0.62f;
	m_tuneStepTrigger = 0.0f;
}

bool StepCycle::isInStance(float p_phi)
{
	// p_phi is always < 1
	float maxt = m_tuneStepTrigger + m_tuneDutyFactor;
	return (maxt <= 1.0f && p_phi >= m_tuneStepTrigger && p_phi < maxt) || // if within bounds, if more than offset and less than offset+len
		(maxt > 1.0f && ((p_phi >= m_tuneStepTrigger) || p_phi < maxt - 1.0f)); // if phase shifted out of bounds(>1), more than offset or less than len-1
}

float StepCycle::getSwingPhase(float p_phi)
{
	// p_phi is always < 1
	if (isInStance(p_phi)) return 0.0f;
	float maxt = m_tuneStepTrigger + m_tuneDutyFactor;
	float pos = p_phi;
	float swinglen = 1.0f - m_tuneDutyFactor; // get total swing time
	if (maxt <= 1.0f) // max is inside bounds
	{
		float rest = 1.0f - maxt; // rightmost rest swing			
		if (p_phi > maxt)
			pos -= maxt; // get start as after end of stance		
		else
			pos += rest; // add rest when at beginning again
		pos /= swinglen; // normalize
		//pos= 1.0f-pos; // invert
	}
	else // max is outside bounds
	{
		float mint = maxt - 1.0f; // start
		pos -= mint;
		pos /= swinglen;
	}
	return pos;
}

float StepCycle::getStancePhase(float p_phi)
{
	// p_t is always < 1
	float maxt = m_tuneStepTrigger + m_tuneDutyFactor;
	if (maxt <= 1.0f && p_phi >= m_tuneStepTrigger && p_phi < maxt)// if within bounds, if more than offset and less than offset+len
	{
		return (p_phi - m_tuneStepTrigger) / m_tuneDutyFactor;
	}
	else if (maxt > 1.0f) // if phase shifted out of bounds(>1), more than offset or less than len-1
	{
		if (p_phi >= m_tuneStepTrigger)
			return (p_phi - m_tuneStepTrigger) / m_tuneDutyFactor;
		else if (p_phi < maxt - 1.0f)
			return (p_phi + 1.0f - m_tuneStepTrigger) / m_tuneDutyFactor;
	}
	return 0.0f;
}

void StepCycle::sanitize()
{
	if (m_tuneDutyFactor > 0.9999f)
	{
		//Debug.Log("df: " + m_tuneDutyFactor);
		m_tuneDutyFactor = 0.9999f;
	}
	if (m_tuneStepTrigger > 0.9999f)
	{
		//Debug.Log("st: " + m_tuneStepTrigger);
		m_tuneStepTrigger = 0.9999f;
	}

	if (m_tuneDutyFactor < 0.00001f)
		m_tuneDutyFactor = 0.00001f;
	if (m_tuneStepTrigger < 0.0f)
		m_tuneStepTrigger = 0.0f;
}

std::vector<float> StepCycle::getParams()
{
	//DEBUGPRINT(("STEP CYCLE GETPARAMS\n"));
	std::vector<float> params;
	params.push_back(m_tuneDutyFactor);
	params.push_back(m_tuneStepTrigger);
	return params;
}

void StepCycle::consumeParams(std::vector<float>& p_other)
{
	OptimizableHelper::ConsumeParamsTo(p_other, &m_tuneDutyFactor);
	OptimizableHelper::ConsumeParamsTo(p_other, &m_tuneStepTrigger);
}

std::vector<float> StepCycle::getParamsMax()
{
	std::vector<float> paramsmax;
	paramsmax.push_back(0.999f); // DF
	paramsmax.push_back(0.999f); // ST
	return paramsmax;
}

std::vector<float> StepCycle::getParamsMin()
{
	std::vector<float> paramsmin;
	paramsmin.push_back(0.0f); // DF
	paramsmin.push_back(0.0f); // ST
	return paramsmin;
}
