#include "Random.h"
#include <vector>

Random::Random()
{

}

Random::~Random()
{

}

float Random::getReal(float p_min, float p_max, Generator p_generator/*= Generator::DETERMINISTIC*/)
{
	return 0.0f; // foo
}

std::vector<float> Random::getRealUniform(float p_min, float p_max, unsigned int p_intervals, Generator p_generator/*= Generator::DETERMINISTIC*/)
{
	return std::vector<float>(); // foo
}

int Random::getInt(int p_min, int p_max, Generator p_generator/*= Generator::DETERMINISTIC*/)
{
	return 0; // foo
}

