#include "Random.h"
#include <vector>
#include <ctime>

Random::Random()
{
	m_detgenerator.seed(c_detseed);
	m_nondetgenerator.seed(time(NULL));
}

Random::~Random()
{

}

float Random::getRealNormal(float p_min, float p_max, Generator p_generator/*= Generator::DETERMINISTIC*/)
{
	std::normal_distribution<float> normal(p_min, p_max);
	float res = normal(*getEnginebyType(p_generator));
	return res;
}

float Random::getRealUniform(float p_min, float p_max, Generator p_generator /*= Generator::DETERMINISTIC*/)
{
	std::uniform_real_distribution<float> uniform(p_min, p_max);
	float res = uniform(*getEnginebyType(p_generator));
	return res;
}

std::vector<float> Random::getRealUniformList(float p_min, float p_max, unsigned int p_population, Generator p_generator/*= Generator::DETERMINISTIC*/)
{
	std::vector<float> res(p_population);
	std::uniform_real_distribution<float> uniform(p_min, p_max);
	std::default_random_engine* engine = getEnginebyType(p_generator);
	for (int i = 0; i < p_population; i++)
		res[i] = uniform(*engine);
	return res; // foo
}

std::vector<double> Random::getRealUniformList(double p_min, double p_max, unsigned int p_population, Generator p_generator/*= Generator::DETERMINISTIC*/)
{
	std::vector<double> res(p_population);
	std::uniform_real_distribution<double> uniform(p_min, p_max);
	std::default_random_engine* engine = getEnginebyType(p_generator);
	for (int i = 0; i < p_population; i++)
		res[i] = uniform(*engine);
	return res; // foo
}


std::default_random_engine* Random::getEnginebyType(Generator p_type)
{
	if (p_type == Generator::DETERMINISTIC)
		return &m_detgenerator;
	else
		return &m_nondetgenerator;
}

