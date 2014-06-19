#pragma once
#include <glm\gtc\type_ptr.hpp>
#include <algorithm>
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
	}
	PDn(float p_Kp, float p_Kd)
	{
		setK(p_Kp, p_Kd);
	}
	~PDn()
	{
		delete[] m_P;
		delete[] m_D;
	}

	float getKp() { return m_Kp; }
	float getKd() { return m_Kd; }
	float setK(float p_Kp, float p_Kd)
	{
		m_Kp = p_Kp; m_Kd = p_Kd;
	}
	float setKp(float p_Kp) { m_Kp = p_Kp; }
	float setKd(float p_Kd) { m_Kd = p_Kd; }

	glm::vec3 getP() { return glm::vec3(m_P[0], m_P[1], m_P[2]); }
	glm::vec3 getD() { return glm::vec3(m_D[0], m_D[1], m_D[2]); }

	// Drive the controller and get new value
	// p_error This is the current error
	// p_dt this is the step size
	float* drive(float* p_error, float p_dt)
	{
		float* res = p_error;
		for (int i = 0; i < c_size; i++)
		{
			float oldError = m_P[i];
			m_P[i] = p_error[i]; // store current error
			//m_I[i] += m_P[i] * p_dt;  // accumulate error velocity to integral term
			m_D[i] = (m_P[i] - oldError) / std::max(0.001f, p_dt); // calculate speed of error change
			// return weighted sum
			res[i] = m_Kp * m_P[i]/* + m_Ki * m_I[i]*/ + m_Kd * m_D[i];
		}
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
			glm::q
			error.ToAngleAxis(out a, out dir);
			// Get torque
			m_vec = drive(a * dir, Time.deltaTime);
		}
		return m_vec; // Note, these are 3 PIDs
	}

	glm::vec2 drive(Vector2 p_error, float p_dt)
	{
		float[] res = { p_error.x, p_error.y };
		res = drive(res, p_dt);
		return new Vector2(res[0], res[1]);
	}

	glm::vec3 drive(Vector3 p_error, float p_dt)
	{
		float[] res = { p_error.x, p_error.y, p_error.z };
		res = drive(res, p_dt);
		return new Vector3(res[0], res[1], res[2]);
	}

	glm::vec4 drive(Vector4 p_error, float p_dt)
	{
		float[] res = { p_error.x, p_error.y, p_error.z, p_error.w };
		res = drive(res, p_dt);
		return new Vector4(res[0], res[1], res[2], res[3]);
	}

protected:
	void initErrorArrays()
	{
		m_P = new float[3];
		m_D = new float[3];
	}
private:
	float m_Kp; // Proportional coefficient
	float m_Kd; // Derivative coefficient

	float* m_P;  // Proportional error (Current error)
	float* m_D;  // Derivative error   (How fast the P error is changing)
};