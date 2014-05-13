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
    public string NAME = "PID";
    public float m_Kp=1.0f; // Proportional coefficient
    public float m_Ki=0.0f; // Integral coefficient
    public float m_Kd=0.1f; // Derivative coefficient

    public float[] m_P = new float[3];  // Proportional error (Current error)
    public float[] m_I = new float[3];  // Integral error     (What we should have corrected before)
    public float[] m_D = new float[3];  // Derivative error   (How fast the P error is changing)

    public static bool m_autoKd = true;

    // Temp:
    // Store result
    public Vector3 m_vec;

    public void Start()
    {
        if (m_autoKd) { m_Kp = 10.0f; m_Kd = 0.1f * m_Kp; }
            //2.0f * Mathf.Sqrt(m_Kp);
    }

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
            m_D[i] = (m_P[i] - oldError) / Mathf.Max(0.001f,p_dt); // calculate speed of error change
            if (float.IsNaN(m_D[i]) || float.IsNaN(m_P[i])) 
                    Debug.Log("error: "+m_P[i] + " olderror: " + oldError + " dt: " + p_dt);       
            // return weighted sum
            res[i] = m_Kp * m_P[i] + m_Ki * m_I[i] + m_Kd * m_D[i];
        }
        return res;
    }

    public Vector3 drive(Quaternion p_current, Quaternion p_goal, float p_dt)
    {
        // To get quaternion "delta", rotate by the inverse of current
        // to get to the origin, then multiply by goal rotation to get "what's left"
        // The resulting quaternion is the "delta".
        Quaternion error = p_goal * Quaternion.Inverse(p_current);
        // Separate angle and axis, so we can feed the axis-wise
        // errors to the PIDs.
        Quaternion ri=Quaternion.identity; ri.w*=-1.0f;
        // If quaternion is not a rotation
        if (error == Quaternion.identity || error == ri)
        {
            m_vec = drive(Vector3.zero, Time.deltaTime);
        }
        else
        {
            float a;
            Vector3 dir;
            error.ToAngleAxis(out a, out dir);
            for (int i = 0; i < m_P.Length; i++)
            {
                bool isnan = false;
                if (float.IsNaN(dir[i]))
                {
                    Debug.Log(" dir is NaN. x" + dir.x + " y" + dir.y + " z" + dir.z);
                    isnan = true;
                }
                if (float.IsNaN(a))
                {
                    Debug.Log(" a is NaN. a" + a);isnan = true;
                }
                float ad = dir[i]*a;
                if (float.IsNaN(dir[i] * a))
                {
                    Debug.Log(" mul is NaN. " + ad);isnan = true;
                }
                if (isnan)
                    Debug.Log("Orig quat="+p_current.ToString()+" inverted="+Quaternion.Inverse(p_current).ToString()+" err="+error.ToString());
            }
            // Get torque
            m_vec = drive(a * dir, Time.deltaTime);
        }
        return m_vec; // Note, these are 3 PIDs
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
