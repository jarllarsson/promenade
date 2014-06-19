#pragma once
#include <glm\gtc\type_ptr.hpp>
#include <exception>
#include "CMatrix.h"
#include "ColorPalettes.h"

// =======================================================================================
//                                      MathHelp
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # MathHelp
/// 
/// 24-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------
#define PI 3.141592653589793238462643383279502884197169399375105820
#define HALFPI 0.5*PI
#define TWOPI 2.0*PI
#define TORAD PI/180
#define TODEG 180/PI
#define PIOVER180 TORAD

namespace MathHelp
{
	size_t roundup(int group_size, int global_size);

	void decomposeTRS(const glm::mat4& m, glm::vec3& scaling,
		glm::mat4& rotation, glm::vec3& translation);


	glm::vec3 transformDirection(const glm::mat4& m, const glm::vec3& p_dir);
	glm::vec3 transformPosition(const glm::mat4& m, const glm::vec3& p_pos);

	glm::vec3 toVec3(const glm::vec4& p_v);
	glm::vec3 toVec3(const Color3f& p_col);
	glm::vec3 getMatrixTranslation(const glm::mat4& m);
	glm::quat getMatrixRotation(const glm::mat4& m);

	float flerp(float p_a, float p_b, float p_t);
	double dlerp(double p_a, double p_b, double p_t);

	void quatToAngleAxis(const glm::quat& p_quat, float& p_outAngle, glm::vec3& p_outAxis);
};