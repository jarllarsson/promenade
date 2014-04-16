using UnityEngine;
using System.Collections;

public class FootStrikeChecker : MonoBehaviour 
{
    private bool isOnGround=true;
	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        if (isFootStrike())
            renderer.material.color = Color.blue;
        else
            renderer.material.color = Color.white;
	}

    public bool isFootStrike()
    {
        return isOnGround;
    }

    
    void OnCollisionEnter()
    {
        isOnGround = true;
    }

    void OnCollisionExit()
    {
        isOnGround = false;
    }
}
