#pragma once
#include <random>
// =======================================================================================
//                                      Random
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Wrapper for random generators
///        
/// # Random
/// 
/// 18-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class Random
{
public:
	Random();
	virtual ~Random();


	enum Generator
	{
		DETERMINISTIC,
		NON_DETERMINISTIC
	};

	float getReal(float p_min, float p_max, Generator p_generator= Generator::DETERMINISTIC);
	std::vector<float> getRealUniform(float p_min, float p_max, unsigned int p_intervals, Generator p_generator= Generator::DETERMINISTIC);

	int getInt(int p_min, int p_max, Generator p_generator= Generator::DETERMINISTIC);

protected:
private:
};