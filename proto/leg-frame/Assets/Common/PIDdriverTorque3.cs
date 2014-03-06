using UnityEngine;
using System.Collections;

/*  ===================================================================
 *                             PIDdriverTorque3
 *  ===================================================================
 *   Uses 3 PIDs to minimize a quaternion error by 
 *   returning a torque vector.
 *   [Deprecated]
 *   */

public class PIDdriverTorque3 : MonoBehaviour 
{
    public PIDn m_pid;
	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {

	}

    public Vector3 drive(Quaternion p_current,Quaternion p_goal)
    {
        // To get quaternion "delta", rotate by the inverse of current
        // to get to the origin, then multiply by goal rotation to get "what's left"
        // The resulting quaternion is the "delta".
        Quaternion error = p_goal * Quaternion.Inverse(p_current);
        // Separate angle and axis, so we can feed the axis-wise
        // errors to the PIDs.
        float a;
        Vector3 dir;
        error.ToAngleAxis(out a, out dir);
        // Get torque
        Vector3 vec=m_pid.drive(a * dir, Time.deltaTime);
        return vec; // Note, these are 3 PIDs
    }
}
