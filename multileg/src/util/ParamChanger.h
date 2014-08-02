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

	}

	virtual ~ParamChanger()
	{

	}

	std::vector<float> change(const std::vector<float>& p_params,
		const std::vector<float>& p_Pmin, const std::vector<float>& p_Pmax,
		int p_iteration);
private:
	/// <summary>
	/// Selection vector, determines whether the parameter
	/// at this position in the list should be changed.
	/// 20% probability of change.
	/// </summary>
	/// <param name="p_size"></param>
	/// <returns></returns>
	std::vector<float> getS(unsigned int p_size);

	/// <summary>
	/// Generate DeltaP the change vector to be added to old P.
	/// It contains randomly activated slots. 
	/// Not all parameters will thus be changed by this vector.
	/// </summary>
	/// <param name="p_P"></param>
	/// <returns></returns>
	std::vector<double> getDeltaP(const std::vector<float>& p_P,
		const std::vector<float>& p_Pmin, const std::vector<float>& p_Pmax,
		int p_iteration);


	std::pair<float, float> getMinMaxOfList(const std::vector<float>& p_list);


private:
	static Random s_randomEngine; // static to avoid reseeds
};