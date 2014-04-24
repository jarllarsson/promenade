using UnityEngine;
using System.Collections;

public class constantmover : MonoBehaviour {
    public float m_speed = 0.1f;
	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
        transform.position += Vector3.forward * m_speed * Time.deltaTime;
	}
}
