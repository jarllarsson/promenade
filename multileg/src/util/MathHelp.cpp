#include "MathHelp.h"

size_t MathHelp::roundup(int group_size, int global_size)
{
	int r = global_size % group_size;
	if (r == 0)
	{
		return global_size;
	}
	else
	{
		return global_size + group_size - r;
	}
}

/**
* Decomposes matrix M such that T * R * S = M, where T is translation matrix,
* R is rotation matrix and S is scaling matrix.
* http://code.google.com/p/assimp-net/source/browse/trunk/AssimpNet/Matrix4x4.cs
* (this method is exact to at least 0.0001f)
*
* | 1  0  0  T1 | | R11 R12 R13 0 | | a 0 0 0 |   | aR11 bR12 cR13 T1 |
* | 0  1  0  T2 |.| R21 R22 R23 0 |.| 0 b 0 0 | = | aR21 bR22 cR23 T2 |
* | 0  0  0  T3 | | R31 R32 R33 0 | | 0 0 c 0 |   | aR31 bR32 cR33 T3 |
* | 0  0  0   1 | |  0   0   0  1 | | 0 0 0 1 |   |  0    0    0    1 |
*
* @param m (in) matrix to decompose
* @param scaling (out) scaling vector
* @param rotation (out) rotation matrix
* @param translation (out) translation vector
*/
void MathHelp::decomposeTRS(const glm::mat4& m, glm::vec3& scaling,
	glm::mat4& rotation, glm::vec3& translation)
{
	// Extract the translation
	translation.x = m[3][0];
	translation.y = m[3][1];
	translation.z = m[3][2];

	// Extract col vectors of the matrix
	glm::vec3 col1(m[0][0], m[0][1], m[0][2]);
	glm::vec3 col2(m[1][0], m[1][1], m[1][2]);
	glm::vec3 col3(m[2][0], m[2][1], m[2][2]);

	//Extract the scaling factors
	scaling.x = glm::length(col1);
	scaling.y = glm::length(col2);
	scaling.z = glm::length(col3);

	// Handle negative scaling
	if (glm::determinant(m) < 0) {
		scaling.x = -scaling.x;
		scaling.y = -scaling.y;
		scaling.z = -scaling.z;
	}

	// Remove scaling from the matrix
	if (scaling.x != 0) {
		col1 /= scaling.x;
	}

	if (scaling.y != 0) {
		col2 /= scaling.y;
	}

	if (scaling.z != 0) {
		col3 /= scaling.z;
	}

	rotation[0][0] = col1.x;
	rotation[0][1] = col1.y;
	rotation[0][2] = col1.z;
	rotation[0][3] = 0.0;

	rotation[1][0] = col2.x;
	rotation[1][1] = col2.y;
	rotation[1][2] = col2.z;
	rotation[1][3] = 0.0;

	rotation[2][0] = col3.x;
	rotation[2][1] = col3.y;
	rotation[2][2] = col3.z;
	rotation[2][3] = 0.0;

	rotation[3][0] = 0.0;
	rotation[3][1] = 0.0;
	rotation[3][2] = 0.0;
	rotation[3][3] = 1.0;
}

glm::vec3 MathHelp::transformDirection(const glm::mat4& m, const glm::vec3& p_dir)
{
	glm::vec4 vec(p_dir.x, p_dir.y, p_dir.z, 0.0f);
	vec = m*/*glm::normalize(*/vec/*)*/;
	return glm::vec3(vec.x,vec.y,vec.z);
}

glm::vec3 MathHelp::transformPosition(const glm::mat4& m, const glm::vec3& p_pos)
{
	glm::vec4 vec(p_pos.x, p_pos.y, p_pos.z, 1.0f);
	vec = m*vec;
	return glm::vec3(vec.x, vec.y, vec.z);
}

glm::vec3 MathHelp::toVec3(const glm::vec4& p_v)
{
	return glm::vec3(p_v.x, p_v.y, p_v.z);
}

glm::vec3 MathHelp::toVec3(const Color3f& p_col)
{
	return glm::vec3(p_col.r, p_col.g, p_col.b);
}

glm::vec3 MathHelp::getMatrixTranslation(const glm::mat4& m)
{
	return MathHelp::toVec3(m * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
}


glm::quat MathHelp::getMatrixRotation(const glm::mat4& m)
{
	return glm::quat(m);
}

float MathHelp::flerp(float p_a, float p_b, float p_t)
{
	return p_b*p_t + p_a*(1.0f-p_t);
}

double MathHelp::dlerp(double p_a, double p_b, double p_t)
{
	return p_b*p_t + p_a*(1.0-p_t);
}

// Extract axis and angle from a quaternion.
// Using the solution provided in:
// http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
//
void MathHelp::quatToAngleAxis(const glm::quat& p_quat, float& p_outAngle, glm::vec3& p_outAxis)
{
	glm::quat q1 = p_quat;
	if (p_quat.w > 1) q1=glm::normalize(p_quat); // if w>1 acos and sqrt will produce errors, this cant happen if quaternion is normalised
	p_outAngle = 2.0f * acos(q1.w);
	double s = sqrt(1.0 - (double)q1.w * (double)q1.w); // assuming quaternion normalised then w is less than 1, so term always positive.
	if (s < 0.001)
	{ // test to avoid divide by zero, s is always positive due to sqrt
		// if s close to zero then direction of axis not important
		p_outAxis.x = q1.x; // if it is important that axis is normalised then replace with x=1; y=z=0;
		p_outAxis.y = q1.y;
		p_outAxis.z = q1.z;
	}
	else
	{
		p_outAxis.x = (float)((double)q1.x / s); // normalise axis
		p_outAxis.y = (float)((double)q1.y / s);
		p_outAxis.z = (float)((double)q1.z / s);
	}
}

glm::vec3 MathHelp::invTransformDirection(const glm::mat4& m, const glm::vec3& p_dir)
{
	return transformDirection(glm::inverse(m), p_dir);
}

glm::vec3 MathHelp::invTransformPosition(const glm::mat4& m, const glm::vec3& p_dir)
{
	return transformPosition(glm::inverse(m), p_dir);
}

// Safe atan2 hahaha
float MathHelp::satan2(float p_x, float p_y)
{
	float a = 0.0f;
	if (p_x != 0.0f && p_y != 0.0f)
		a = atan2(p_x, p_y); // range [-pi, +pi] radians.
	return a;
}

