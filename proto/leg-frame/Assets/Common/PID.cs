using UnityEngine;
using System.Collections;

/*  ===================================================================
 *                     A generic PID-controller
 *  ===================================================================
 *   Controller logic for error minimization.
 *   */
public class PID : MonoBehaviour 
{
    public float m_Kp=1.0f; // Proportional coefficient
    public float m_Ki=0.0f; // Integral coefficient
    public float m_Kd=0.1f; // Derivative coefficient

    public float m_P = 0.0f;  // Proportional error (Current error)
    public float m_I = 0.0f;  // Integral error     (What we should have corrected before)
    public float m_D = 0.0f;  // Derivative error   (How fast the P error is changing)

    // Drive the controller and get new value
    // p_error This is the current error
    // p_dt this is the step size
    public float drive(float p_error, float p_dt)
    {
        float oldError = m_P;
        m_P = p_error; // store current error
        m_I += m_P * p_dt;  // accumulate error velocity to integral term
        m_D = (m_P - oldError) / p_dt; // calculate speed of error change
        // return weighted sum
        return m_Kp * m_P + m_Ki * m_I + m_Kd * m_D;
    }

}
