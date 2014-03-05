using UnityEngine;
using System.Collections;

/*  ===================================================================
 *                     A generic PID-controller
 *                      for n-scalar errors
 *  ===================================================================
 *   Wraps several PIDs into one for minimizing several scalars using
 *   the same coefficients. Useful for vectors for example.
 *   */
public class PIDn : MonoBehaviour 
{
    public float m_Kp=1.0f; // Proportional coefficient
    public float m_Ki=0.0f; // Integral coefficient
    public float m_Kd=0.1f; // Derivative coefficient

    public float[] m_P = new float[3];  // Proportional error (Current error)
    public float[] m_I = new float[3];  // Integral error     (What we should have corrected before)
    public float[] m_D = new float[3];  // Derivative error   (How fast the P error is changing)

    // Drive the controller and get new value
    // p_error This is the current error
    // p_dt this is the step size
    public float[] drive(float[] p_error, float p_dt)
    {
        float[] res = p_error;
        for (int i=0;i<m_P.Length;i++)
        {
            float oldError = m_P[i];
            m_P[i] = p_error[i]; // store current error
            m_I[i] += m_P[i] * p_dt;  // accumulate error velocity to integral term
            m_D[i] = (m_P[i] - oldError) / p_dt; // calculate speed of error change
            // return weighted sum
            res[i] = m_Kp * m_P[i] + m_Ki * m_I[i] + m_Kd * m_D[i];
        }
        return res;
    }

    public Vector2 drive(Vector2 p_error, float p_dt)
    {
        float[] res = { p_error.x, p_error.y};
        res = drive(res, p_dt);
        return new Vector2(res[0], res[1]);
    }

    public Vector3 drive(Vector3 p_error, float p_dt)
    {
        float[] res = {p_error.x,p_error.y,p_error.z};
        res = drive(res, p_dt);
        return new Vector3(res[0], res[1], res[2]);
    }

    public Vector4 drive(Vector4 p_error, float p_dt)
    {
        float[] res = { p_error.x, p_error.y, p_error.z, p_error.w };
        res = drive(res, p_dt);
        return new Vector4(res[0], res[1], res[2], res[3]);
    }

}
