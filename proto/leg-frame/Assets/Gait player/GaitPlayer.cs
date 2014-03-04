using UnityEngine;
using System.Collections;

public class GaitPlayer : MonoBehaviour 
{
    // Total gait time (ie. stride duration)
    public float m_tuneGaitPeriod=1.0f; // T

    // Current gait phase
    public float m_gaitPhase=0.0f; // phi

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        updatePhase(Time.deltaTime);
	}

    void updatePhase(float p_t)
    {
        m_gaitPhase += p_t / m_tuneGaitPeriod;
        while (m_gaitPhase > 1.0f)
            m_gaitPhase -= 1.0f;
    }
}
