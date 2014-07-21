#pragma once
#include <vector>
#include <glm\gtc\type_ptr.hpp>

// =======================================================================================
//                                      OptimizableHelper
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Helper functions for IOptimizables
///        
/// # OptimizableHelper
/// 
/// 16-7-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

namespace OptimizableHelper
{
	void popFront(std::vector<float>& p_params, unsigned int p_range = 1);

	std::vector<float> ExtractParamsListFrom(const glm::vec2& p_vec2);

	std::vector<float> ExtractParamsListFrom(const glm::vec3& p_vec3);

	std::vector<float> ExtractParamsListFrom(const glm::quat& p_quat);

	std::vector<float> ExtractParamsListFrom(const glm::vec4& p_vec4);

	void ConsumeParamsTo(std::vector<float>& p_params, float* p_inoutFloat);

	void ConsumeParamsTo(std::vector<float>& p_params, glm::vec2* p_inoutVec2);

	void ConsumeParamsTo(std::vector<float>& p_params, glm::vec3* p_inoutVec3);

	void ConsumeParamsTo(std::vector<float>& p_params, glm::vec4* p_inoutVec4);

	void ConsumeParamsTo(std::vector<float>& p_params, glm::quat* p_inoutQuat);

	void addRange(std::vector<float>& p_params, const std::vector<float>& p_rangeToAdd);
};