using UnityEngine;
using System.Collections;

/*  ===================================================================
 *                          PIDdriverScalar 
 *  ===================================================================
 *   Uses a PID to minimize a scalar error on the set parameter.
 *   */

public class PIDdriverScalar : MonoBehaviour 
{
    public PID m_pid;
    public float goal = 1000.0f;
    public float current = 0.0f;
	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        current += m_pid.drive(goal - current, Time.deltaTime)*Time.deltaTime;
	}
}
