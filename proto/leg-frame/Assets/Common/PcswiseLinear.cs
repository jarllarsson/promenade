using UnityEngine;
using System.Collections;
using UnityEditor;
using System;
using System.Collections.Generic;

/*  ===================================================================
 *                     Piece wise linear function
 *  ===================================================================
 *   Function defined by data points and linear interpolation.
 */

public class PcswiseLinear : MonoBehaviour, IOptimizable
{
    public string NAME = "PL";
    // Type to be inited as
    public enum INITTYPE
    {
        SIN,
        COS,
        COS_INV_NORM, // inverted and normalized cos
        HALF_SIN,      // half sine
        FLAT,        // flat zero
        LIN_INC,    // Linear increase
        RND,
        LIN_DEC
    }

    // Number of data points
    public static int s_size = 30;

    public Color m_dbgDrawCol = Color.green;

    // The data points of the function
    private float[] m_tuneDataPoints = new float[s_size];

    // IOptimizable
    public List<float> GetParams()
    {
        return new List<float>(m_tuneDataPoints);
        // Here one could also implement setters for children IOptimizables
    }

    public void ConsumeParams(List<float> p_params)
    {
        m_tuneDataPoints = p_params.ToArray();
        // Here one could also implement getters for children IOptimizables
    }


    /// /////////////////////////////////////////////////////////////
    /// /////////////////////////////////////////////////////////////

    public INITTYPE m_initAsFunc = INITTYPE.COS_INV_NORM;


    // Editor specific to show curve
    public AnimationCurve m_curve = AnimationCurve.Linear(0.0f,0.0f,1.0f,1.0f);
    // Debugging stuff
    public float m_resetScale = 1;
    public bool m_reset = false;

    void Awake()
    {
        if (m_reset)
        {
            reset(m_resetScale);
            m_reset = false;
        }

    }

    public void reset(float p_scale=1.0f)
    {
        for (int i = 0; i < s_size; i++)
        {
            float t = getTimeFromIdx(i);
            switch (m_initAsFunc)
            {
                case INITTYPE.SIN:
                    m_tuneDataPoints[i] = p_scale*Mathf.Sin(t * 2.0f * Mathf.PI);
                    break;
                case INITTYPE.COS:
                    m_tuneDataPoints[i] = p_scale * Mathf.Cos(t * 2.0f * Mathf.PI);
                    break;
                case INITTYPE.COS_INV_NORM:
                    m_tuneDataPoints[i] = p_scale * ((Mathf.Cos(t * 2.0f * Mathf.PI) - 1.0f) * -0.5f);
                    break;
                case INITTYPE.HALF_SIN:
                    m_tuneDataPoints[i] = p_scale * Mathf.Sin(t * Mathf.PI);
                    break;
                case INITTYPE.FLAT:
                    m_tuneDataPoints[i] = 0.0f;
                    break;
                case INITTYPE.LIN_INC:
                    m_tuneDataPoints[i] = p_scale * t;
                    break;
                case INITTYPE.LIN_DEC:
                    m_tuneDataPoints[i] = p_scale * (1.0f - t);
                    break;
                case INITTYPE.RND:
                    m_tuneDataPoints[i] = UnityEngine.Random.Range(-1.0f, 1.0f);
                    break;
                default:
                    m_tuneDataPoints[i] = p_scale * 0.5f;
                    break;
            }
        }

        // Update visualization
        while (m_curve.length > 0) m_curve.RemoveKey(0);
        for (int i = 0; i < s_size; i++)
        {
            float t = getTimeFromIdx(i);
            m_curve.AddKey(t, m_tuneDataPoints[i]);
        }
    }

    float getTimeFromIdx(int p_idx)
    {
        return (float)p_idx / (float)(s_size-1);
    }

	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        if (m_reset)
        {
            reset(m_resetScale);
            m_reset = false;  
        }

        draw(0.0f);
	}

    public void draw(float p_t)
    {
        float s = transform.localScale.x;
        for (int i = 0; i < 100; i++)
        {
            float t = i * 0.01f;
            float t1 = (i + 1) * 0.01f;
            Debug.DrawLine(transform.position + Vector3.right * s * t + Vector3.up * getValAt(t),
                transform.position + Vector3.right * s * t1 + Vector3.up * getValAt(t1),
                m_dbgDrawCol,p_t);
        }
    }

    public float getValAt(float p_phi)
    {
        float realTime = (float)(s_size-1) * p_phi;
        // lower bound idx (never greater than last idx)
        int lowIdx = (int)(realTime) % (s_size);
        // higher bound idx (loops back to 1 if over)
        int hiIdx = ((int)(realTime)+1) % (s_size);
        // get amount of interpolation by subtracting the base from current
        float lin = p_phi * (float)(s_size-1) - (float)lowIdx;
        //Debug.Log(realTime + ": " + lowIdx + "->" + hiIdx + " [t" + lin + "]");
        //Debug.Log(hi);
        float val = 0.0f;
        try
        {
            val = Mathf.Lerp(m_tuneDataPoints[lowIdx], m_tuneDataPoints[hiIdx], lin);
        }
        catch(Exception e)
        {
            Debug.Log(e.Message.ToString());
            Debug.Log(p_phi + ": " + lowIdx + "->" + hiIdx + " [t" + lin + "]");
        }
        return val;
    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.white;
        Gizmos.DrawWireSphere(transform.position, 0.2f);
    }
}
