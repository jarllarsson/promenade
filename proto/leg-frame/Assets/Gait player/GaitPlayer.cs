using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/*  ===================================================================
 *                          Gait player
 *  ===================================================================
 *   Contains data on total time and current time, ie. the timeline.
 *   */
public class GaitPlayer : MonoBehaviour, IOptimizable
{
    // Total gait time (ie. stride duration)
    public float m_tuneGaitPeriod=1.0f; // T

    // Current gait phase
    public float m_gaitPhase=0.0f; // phi

    private bool m_hasRestarted_oneCheck = false; // can be read once after each restart of stride

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () 
    {

	}

    // IOptimizable
    public List<float> GetParams()
    {
        List<float> vals = new List<float>();
        vals.Add(m_tuneGaitPeriod);
        return vals;
    }

    public void ConsumeParams(List<float> p_params)
    {
        OptimizableHelper.ConsumeParamsTo(p_params, ref m_tuneGaitPeriod);
    }

    public List<float> GetParamsMax()
    {
        List<float> maxList = new List<float>();
        maxList.Add(3.0f);
        return maxList;
    }

    public List<float> GetParamsMin()
    {
        List<float> minList = new List<float>();
        minList.Add(0.01f);
        return minList;
    }

    public void updatePhase(float p_t)
    {
        m_gaitPhase += p_t / m_tuneGaitPeriod;
        while (m_gaitPhase > 1.0f)
        {
            m_gaitPhase -= 1.0f;
            m_hasRestarted_oneCheck = true;
        }
    }

    public bool checkHasRestartedStride_AndResetFlag() // ugh...
    {
        bool res = m_hasRestarted_oneCheck;
        m_hasRestarted_oneCheck = false;
        return res;
    }
}
