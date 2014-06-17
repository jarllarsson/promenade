#pragma once

// =======================================================================================
//                                      PID
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	A generic PID-controller (Proportionate-Integral-Derivate)
///        
/// # PID
/// 
/// 17-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class PID
{
public:
	PID()
	{
		m_Kp = 1.0f;
		m_Ki = 0.0f;
		m_Kd = 0.1f;
	}
	PID(float p_Kp, float p_Ki, float p_Kd)
	{
		setK(p_Kp, p_Ki, p_Kd);
	}
	~PID();

	float getKp() {return m_Kp;}
	float getKi() {return m_Ki;}
	float getKd() {return m_Kd;}
	float setK(float p_Kp, float p_Ki, float p_Kd) 
	{ m_Kp = p_Kp; m_Ki = p_Ki; m_Kd = p_Kd; }
	float setKp(float p_Kp) { m_Kp=p_Kp; }
	float setKi(float p_Ki) { m_Ki=p_Ki; }
	float setKd(float p_Kd) { m_Kd=p_Kd; }

	float getP() { return m_P; }
	float getI() { return m_I; }
	float getD() { return m_D; }

	// Drive the controller and get new value
	// p_error This is the current error
	// p_dt this is the step size
	float drive(float p_error, float p_dt)
	{
		float oldError = m_P;
		m_P = p_error; // store current error
		m_I += m_P * p_dt;  // accumulate error velocity to integral term
		m_D = (m_P - oldError) / p_dt; // calculate speed of error change
		// return weighted sum
		return m_Kp * m_P + m_Ki * m_I + m_Kd * m_D;
	}

protected:
private:
	float m_Kp; // Proportional coefficient
	float m_Ki; // Integral coefficient
	float m_Kd; // Derivative coefficient

	float m_P;  // Proportional error (Current error)
	float m_I;  // Integral error     (What we should have corrected before)
	float m_D;  // Derivative error   (How fast the P error is changing)
};