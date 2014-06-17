#pragma once

// =======================================================================================
//                                      PD
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	A generic PID-controller (Proportionate-Integral-Derivate)
///			With no Integral term (removed from data structure as well to minimize size)
///        
/// # PD
/// 
/// 17-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class PD
{
public:
	PD()
	{
		m_Kp = 1.0f;
		m_Kd = 0.1f;
	}
	PD(float p_Kp, float p_Kd)
	{
		setK(p_Kp, p_Kd);
	}
	~PD();

	float getKp() { return m_Kp; }
	float getKd() { return m_Kd; }
	float setK(float p_Kp, float p_Kd)
	{
		m_Kp = p_Kp; m_Kd = p_Kd;
	}
	float setKp(float p_Kp) { m_Kp = p_Kp; }
	float setKd(float p_Kd) { m_Kd = p_Kd; }

	float getP() { return m_P; }
	float getD() { return m_D; }

	// Drive the controller and get new value
	// p_error This is the current error
	// p_dt this is the step size
	float drive(float p_error, float p_dt)
	{
		float oldError = m_P;
		m_P = p_error; // store current error
		m_D = (m_P - oldError) / p_dt; // calculate speed of error change
		// return weighted sum
		return m_Kp * m_P + m_Kd * m_D;
	}

protected:
private:
	float m_Kp; // Proportional coefficient
	float m_Kd; // Derivative coefficient

	float m_P;  // Proportional error (Current error)
	float m_D;  // Derivative error   (How fast the P error is changing)
};