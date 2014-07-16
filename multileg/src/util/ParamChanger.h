#pragma once
#include "Random.h"
#include "ValueClamp.h"
// =======================================================================================
//                                      ParamChanger
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Class that changes a parameter list based on a mask and boundary values
///        
/// # Random
/// 
/// 16-7-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class ParamChanger
{
public:
	ParamChanger()
	{
		m_randomEngine = new Random();
	}

	virtual ~ParamChanger()
	{
		delete m_randomEngine;
	}

	std::vector<float> change(const std::vector<float>& p_params, 
		const std::vector<float>& p_Pmin, const std::vector<float>& p_Pmax, 
		int p_iteration)
	{
		int size = p_params.size();
		std::vector<float> result(size);
		std::vector<double> deltaP = getDeltaP(p_params, 
											   p_Pmin, p_Pmax, p_iteration);
		for (unsigned int i = 0; i < size; i++)
		{
			result[i] = (float)((double)p_params[i] + deltaP[i]);
			result[i] = clamp(result[i], p_Pmin[i], p_Pmax[i]);
		}
		
		return result;
	}
private:
	Random* m_randomEngine;
};