using UnityEngine;
using System.Collections;

public class AngleSetter : MonoBehaviour 
{
    public float speed=10.0f;
	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        float horiz = Input.GetAxis("Horizontal");
        horiz = Mathf.Sin(Time.time*3.0f);
        transform.Rotate(-Vector3.forward,horiz*speed*Time.deltaTime);
	}
}
