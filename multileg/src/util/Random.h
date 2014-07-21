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

	int getRandomInt(int p_min, int p_max, Generator p_generator = Generator::DETERMINISTIC);
	float getRealNormal(float p_min, float p_max, Generator p_generator= Generator::DETERMINISTIC);
	float getRealUniform(float p_min, float p_max, Generator p_generator = Generator::DETERMINISTIC);
	std::vector<float> getRealUniformList(float p_min, float p_max, unsigned int p_population, Generator p_generator= Generator::DETERMINISTIC);
	std::vector<double> getRealUniformList(double p_min, double p_max, unsigned int p_population, Generator p_generator = Generator::DETERMINISTIC);

	
protected:
	std::default_random_engine* getEnginebyType(Generator p_type);
private:
	static const int c_detseed = 4350809;
	std::default_random_engine m_detgenerator;
	std::default_random_engine m_nondetgenerator;
};