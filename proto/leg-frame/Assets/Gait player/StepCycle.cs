using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/*  ===================================================================
 *                          Step cycle
 *  ===================================================================
 *   Data describing movement for one foot in normalized time in 
 *   relation to the overall gait of all limbs.
 *   */

public class StepCycle : MonoBehaviour, IOptimizable
{
    // Fraction of overall normalized time for which the 
    // foot is touching the ground.
    public float m_tuneDutyFactor;

    // Offset time point in normalized time when the
    // foot begins its cycle.
    public float m_tuneStepTrigger;


    // IOptimizable
    public List<float> GetParams()
    {
        List<float> vals = new List<float>();
        vals.Add(m_tuneDutyFactor);
        vals.Add(m_tuneStepTrigger);
        return vals;
    }

    public void ConsumeParams(List<float> p_params)
    {
        m_tuneDutyFactor = p_params[0];
        m_tuneStepTrigger = p_params[1];
        for (int i = 0; i < 2;i++ )
            p_params.RemoveAt(0);
        sanitize();
    }

    public List<float> GetParamsMax()
    {
        List<float> maxList = new List<float>();
        maxList.Add(0.99f);
        maxList.Add(0.99f);
        return maxList;
    }

    public List<float> GetParamsMin()
    {
        List<float> minList = new List<float>();
        minList.Add(0.001f);
        minList.Add(0.001f);
        return minList;
    }

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    void sanitize()
    {
        if (m_tuneDutyFactor > 0.9999f)
        {
            Debug.Log("df: " + m_tuneDutyFactor);
            m_tuneDutyFactor = 0.9999f;
        }
        if (m_tuneStepTrigger > 0.9999f)
        {
            Debug.Log("st: " + m_tuneStepTrigger);
            m_tuneStepTrigger = 0.9999f;
        }

        if (m_tuneDutyFactor < 0.00001f)
            m_tuneDutyFactor = 0.00001f;
        if (m_tuneStepTrigger < 0.0f)
            m_tuneStepTrigger = 0.0f;
    }

    public bool isInStance(float p_phi)
    {
        // p_t is always < 1
        float maxt = m_tuneStepTrigger + m_tuneDutyFactor;
        return (maxt <= 1.0f && p_phi >= m_tuneStepTrigger && p_phi < maxt) || // if within bounds, if more than offset and less than offset+len
               (maxt > 1.0f && ((p_phi >= m_tuneStepTrigger) || p_phi < maxt - 1.0f)); // if phase shifted out of bounds(>1), more than offset or less than len-1
    }

    //
    // Get phase of swing, is zero in stance (0-1-0)
    // can be used as period in transition function
    //
    public float getSwingPhase(float p_phi)
    {
        // p_t is always < 1
        if (isInStance(p_phi)) return 0.0f;
        float maxt = m_tuneStepTrigger + m_tuneDutyFactor;
        float pos = p_phi;
        float swinglen = 1.0f - m_tuneDutyFactor; // get total swing time
        if (maxt <= 1.0f) // max is inside bounds
        {
            float rest = 1.0f - maxt; // rightmost rest swing			
            if (p_phi > maxt)
                pos -= maxt; // get start as after end of stance		
            else
                pos += rest; // add rest when at beginning again
            pos /= swinglen; // normalize
            //pos= 1.0f-pos; // invert
        }
        else // max is outside bounds
        {
            float mint = maxt - 1.0f; // start
            pos -= mint;
            pos /= swinglen;
        }
        return pos;
    }

    public float getStancePhase(float p_phi)
    {
        // p_t is always < 1
        float maxt = m_tuneStepTrigger + m_tuneDutyFactor;
        if (maxt <= 1.0f && p_phi >= m_tuneStepTrigger && p_phi < maxt)// if within bounds, if more than offset and less than offset+len
        { 
            return (p_phi-m_tuneStepTrigger)/m_tuneDutyFactor;
        }
        else if(maxt > 1.0f) // if phase shifted out of bounds(>1), more than offset or less than len-1
        {
            if (p_phi >= m_tuneStepTrigger)
                return (p_phi - m_tuneStepTrigger) / m_tuneDutyFactor;
            else if (p_phi < maxt - 1.0f)
                return (p_phi + 1.0f - m_tuneStepTrigger) / m_tuneDutyFactor;
        }
        return 0.0f;
    }
}
