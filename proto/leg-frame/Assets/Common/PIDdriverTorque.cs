using UnityEngine;
using System.Collections;

/*  ===================================================================
 *                             PIDdriverTorque
 *  ===================================================================
 *   Uses a PID to minimize an angle error by 
 *   returning a torque scalar.
 *   */

public class PIDdriverTorque : MonoBehaviour 
{
    public PID m_pid;
    public Transform goal;
    private float torque = 0.0f;
	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        float current = transform.rotation.eulerAngles.z;
        float goalAngle = goal.rotation.eulerAngles.z;
        torque = m_pid.drive(Mathf.DeltaAngle(current,goalAngle), Time.deltaTime);
	}

    void FixedUpdate()
    {
        rigidbody.AddTorque(transform.forward*torque);
    }
}
