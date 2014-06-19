#pragma once
#include <glm\gtc\type_ptr.hpp>
#include <algorithm>
#include <MathHelp.h>
// =======================================================================================
//                                      PDn
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # PDn
/// 
/// 19-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class PDn
{
public:
	static const unsigned int c_size = 3;
	PDn()
	{
		m_Kp = 1.0f;
		m_Kd = 0.1f;
		initErrorArrays();
	}
	PDn(float p_Kp, float p_Kd)
	{
		setK(p_Kp, p_Kd);
		initErrorArrays();
	}
	~PDn()
	{

	}

	float getKp() { return m_Kp; }
	float getKd() { return m_Kd; }
	void setK(float p_Kp, float p_Kd)
	{
		m_Kp = p_Kp; m_Kd = p_Kd;
	}
	void setKp(float p_Kp) { m_Kp = p_Kp; }
	void setKd(float p_Kd) { m_Kd = p_Kd; }

	glm::vec3 getP() { return glm::vec3(m_P[0], m_P[1], m_P[2]); }
	glm::vec3 getD() { return glm::vec3(m_D[0], m_D[1], m_D[2]); }

	// Drive the controller and get new value
	// p_error This is the current error
	// p_dt this is the step size
	float drive(float p_error, unsigned int p_id, float p_dt)
	{
		float oldError = m_P[p_id];
		m_P[p_id] = p_error; // store current error
		m_D[p_id] = (m_P[p_id] - oldError) / p_dt; // calculate speed of error change
		// return weighted sum
		return m_Kp * m_P[p_id] + m_Kd * m_D[p_id];
	}

	glm::vec2 drive(const glm::vec2& p_error, float p_dt)
	{
		glm::vec2 res( p_error.x, p_error.y );
		for (unsigned int i = 0; i < 2;i++)
			res[i] = drive(res[i],i, p_dt);
		return res;
	}

	glm::vec3 drive(const glm::vec3& p_error, float p_dt)
	{
		glm::vec3 res(p_error.x, p_error.y, p_error.z);
		for (unsigned int i = 0; i < 3; i++)
			res[i] = drive(res[i], i, p_dt);
		return res;
	}


	glm::vec3 drive(const glm::quat& p_current, const glm::quat& p_goal, float p_dt)
	{
		// To get quaternion "delta", rotate by the inverse of current
		// to get to the origin, then multiply by goal rotation to get "what's left"
		// The resulting quaternion is the "delta".
		glm::quat error = glm::inverse(p_current) * p_goal;
		// Separate angle and axis, so we can feed the axis-wise
		// errors to the PIDs.
		glm::quat ri; ri.w *= -1.0f;
		glm::vec3 result;
		// If quaternion is not a rotation
		if (error == glm::quat() || error == ri)
		{
			result = drive(glm::vec3(0.0f), p_dt);
		}
		else
		{
			float a=0.0f;
			glm::vec3 dir;
			MathHelp::quatToAngleAxis(error, a, dir);
			// Get torque
			result = drive(a * dir, p_dt);
		}
		return result; // Note, these are 3 PIDs
	}

protected:
	void initErrorArrays()
	{
		for (int i = 0; i < 3; i++)
		{
			m_P[i] = 0.0f; m_D[i] = 0.0f;
		}
	}
private:
	float m_Kp; // Proportional coefficient
	float m_Kd; // Derivative coefficient

	float m_P[3];  // Proportional error (Current error)
	float m_D[3];  // Derivative error   (How fast the P error is changing)
};