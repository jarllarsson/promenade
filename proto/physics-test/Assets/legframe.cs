using UnityEngine;
using System.Collections;

public class legframe : MonoBehaviour {
	public ConstantForce mForce;
	public BoxCollider mColl;
	public float mSpd=3.0f;
	public float mLen = 1.0f;
	public float mHt = 1.0f;
	public float mSign=1.0f;
	public float mMP=1.0f;
	private Vector3 t;
	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void FixedUpdate () 
	{
		t = new Vector3((1.0f+Mathf.Sin(Time.time*mSpd)*mSign)*0.5f*mMP,0.0f,0.0f);
		//mForce.relativeTorque = new Vector3(Mathf.Sin(Time.time*mSpd)*mSign*mMP,0.0f,0.0f);		
		rigidbody.AddRelativeTorque(t);
		mColl.center = new Vector3(0.0f,-mLen+(1.0f+Mathf.Cos(Time.time*mSpd)*-mSign)*0.5f*mHt,0.0f);
		Debug.DrawLine(transform.position,transform.position+transform.parent.localScale.y*transform.TransformDirection(mColl.center),Color.green,0.05f);
	}
	
	void OnDrawGizmos()
	{
		Gizmos.color = Color.blue;
		Vector3 line=new Vector3(0.0f,0.0f,t.x);
		//Gizmos.DrawLine(transform.position, transform.position+transform.TransformDirection(line));
	}
}
