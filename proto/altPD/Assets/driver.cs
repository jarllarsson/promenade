using UnityEngine;
using System.Collections;

public class driver : MonoBehaviour 
{
	public PIDn m_driver;
	public Transform m_goal;
	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void FixedUpdate () 
	{
		rigidbody.AddTorque(m_driver.drive(transform.rotation,m_goal.rotation,Time.deltaTime));
	}
}
