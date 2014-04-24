using UnityEngine;
using System.Collections;

public class FootStrikeChecker : MonoBehaviour 
{
    private bool isOnGround=false;
	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        if (isFootStrike())
            renderer.material.color += Color.blue*0.3f;
        else
            renderer.material.color += Color.white*0.3f;
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
