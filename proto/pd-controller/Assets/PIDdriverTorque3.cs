using UnityEngine;
using System.Collections;

public class PIDdriverTorque3 : MonoBehaviour 
{
    public PIDn m_pid;
    public Transform goal;
    private Vector3 torque;
	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        // To get quaternion "delta", rotate by the inverse of current
        // to get to the origin, then multiply by goal rotation to get "what's left"
        // The resulting quaternion is the "delta".
        Quaternion error = goal.rotation * Quaternion.Inverse(transform.rotation);
        //Debug.Log(error.ToString());
        //transform.rotation *= error;
        float a;
        Vector3 dir; 
        error.ToAngleAxis(out a, out dir);
        // Get scalars
        //float x, y, z;
        //x = m_pidX.drive(a*dir.x, Time.deltaTime);
        //y = m_pidY.drive(a*dir.y, Time.deltaTime);
        //z = m_pidZ.drive(a*dir.z, Time.deltaTime);
        torque = m_pid.drive(a * dir, Time.deltaTime);
        //
        //torque = new Vector3(x, y, z);
        Debug.Log(torque.ToString());
	}

    void FixedUpdate()
    {
        rigidbody.AddTorque(torque);
    }
}
