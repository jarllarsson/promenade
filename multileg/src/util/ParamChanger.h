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
	/// <summary>
	/// Selection vector, determines whether the parameter
	/// at this position in the list should be changed.
	/// 20% probability of change.
	/// </summary>
	/// <param name="p_size"></param>
	/// <returns></returns>
	std::vector<float> getS(unsigned int p_size)
	{
		int changeProbabilityPercent = 20;
		std::vector<float> S(p_size);
		for (int i = 0; i < p_size; i++)
		{
			S[i] = m_randomEngine->getInt(0, 99) < changeProbabilityPercent ? 1.0f : 0.0f;
		}
		return S;
	}

	/// <summary>
	/// Generate DeltaP the change vector to be added to old P.
	/// It contains randomly activated slots. 
	/// Not all parameters will thus be changed by this vector.
	/// </summary>
	/// <param name="p_P"></param>
	/// <returns></returns>
	std::vector<double> getDeltaP(const std::vector<float>& p_P,
		const std::vector<float>& p_Pmin, const std::vector<float>& p_Pmax,
		int p_iteration)
	{
		int size = p_P.size();

		// Get S vector
		std::vector<float> S = getS(size);

		// Calculate delta-P
		std::vector<double> deltaP(size);
		std::vector<double> U = m_randomEngine->getRealUniform(-0.1,0.1,size);
		for (unsigned int i = 0; i < size; i++)
		{
			double P = (double)p_P[i];
			double R = p_Pmax[i] - p_Pmin[i];
			double c = U[i]*R;

			deltaP[i] = (double)S[i] * c;
		}
		return deltaP;
	}

	std::pair<float,float> getMinMaxOfList(const std::vector<float>& p_list)
	{
		float valuemin = 999999999.0f;
		float valuemax = 0.0f;
		for (unsigned int i = 0; i<(unsigned int)p_list.size(); i++)
		{
			if (p_list[i] > valuemax) valuemax = p_list[i];
			if (p_list[i] < valuemin) valuemin = p_list[i];
		}
		return pair<float, float>(valuemin, valuemax);
	}


private:
	Random* m_randomEngine;
};