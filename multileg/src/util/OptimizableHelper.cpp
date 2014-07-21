#include "OptimizableHelper.h"


void OptimizableHelper::popFront(std::vector<float>& p_params, unsigned int p_range/* = 1*/)
{
	p_params = std::vector<float>(p_params.begin() + p_range,
		p_params.end());
}

std::vector<float> OptimizableHelper::ExtractParamsListFrom(const glm::vec2& p_vec2)
{
	std::vector<float> vals;
	vals.push_back(p_vec2.x);
	vals.push_back(p_vec2.y);
	return vals;
}

std::vector<float> OptimizableHelper::ExtractParamsListFrom(const glm::vec3& p_vec3)
{
	std::vector<float> vals;
	vals.push_back(p_vec3.x);
	vals.push_back(p_vec3.y);
	vals.push_back(p_vec3.z);
	return vals;
}

std::vector<float> OptimizableHelper::ExtractParamsListFrom(const glm::quat& p_quat)
{
	std::vector<float> vals;
	vals.push_back(p_quat.x);
	vals.push_back(p_quat.y);
	vals.push_back(p_quat.z);
	vals.push_back(p_quat.w);
	return vals;
}

std::vector<float> OptimizableHelper::ExtractParamsListFrom(const glm::vec4& p_vec4)
{
	std::vector<float> vals;
	vals.push_back(p_vec4.x);
	vals.push_back(p_vec4.y);
	vals.push_back(p_vec4.z);
	vals.push_back(p_vec4.w);
	return vals;
}

void OptimizableHelper::ConsumeParamsTo(std::vector<float>& p_params, float* p_inoutFloat)
{
	*p_inoutFloat = p_params[0];
	popFront(p_params);
}

void OptimizableHelper::ConsumeParamsTo(std::vector<float>& p_params, glm::vec2* p_inoutVec2)
{
	for (int i = 0; i < 2; i++)
	{
		(*p_inoutVec2)[i] = p_params[i];
	}
	popFront(p_params, 2);
}

void OptimizableHelper::ConsumeParamsTo(std::vector<float>& p_params, glm::vec3* p_inoutVec3)
{
	for (int i = 0; i < 3; i++)
	{
		(*p_inoutVec3)[i] = p_params[i];
	}
	popFront(p_params, 3);
}

void OptimizableHelper::ConsumeParamsTo(std::vector<float>& p_params, glm::vec4* p_inoutVec4)
{
	for (int i = 0; i < 4; i++)
	{
		(*p_inoutVec4)[i] = p_params[i];
	}
	popFront(p_params, 4);
}


void OptimizableHelper::ConsumeParamsTo(std::vector<float>& p_params, glm::quat* p_inoutQuat)
{
	for (int i = 0; i < 4; i++)
	{
		(*p_inoutQuat)[i] = p_params[i];
	}
	popFront(p_params, 4);
}

void OptimizableHelper::addRange(std::vector<float>& p_params, const std::vector<float>& p_rangeToAdd)
{
	p_params.insert(p_params.end(), p_rangeToAdd.begin(), p_rangeToAdd.end());
}