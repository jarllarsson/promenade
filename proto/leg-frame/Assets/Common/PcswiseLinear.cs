using UnityEngine;
using System.Collections;
using UnityEditor;
using System;

/*  ===================================================================
 *                     Piece wise linear function
 *  ===================================================================
 *   Function defined by data points and linear interpolation.
 */

public class PcswiseLinear : MonoBehaviour // extends editor only to visualize graph
{
    // Type to be inited as
    public enum INITTYPE
    {
        SIN,
        COS,
        COS_INV_NORM, // inverted and normalized cos
        HALF_SIN      // half sine
    }

    // Number of data points
    public static int s_size = 6;

    // The data points of the function
    public float[] m_tuneDataPoints = new float[s_size];

    public INITTYPE m_initAsFunc = INITTYPE.COS_INV_NORM;


    // Editor specific to show curve
    public AnimationCurve m_curve = AnimationCurve.Linear(0.0f,0.0f,1.0f,1.0f);

    void Awake()
    {
        for (int i=0;i<s_size;i++)
        {
            float t=getTimeFromIdx(i);
            switch(m_initAsFunc)
            {
                case INITTYPE.SIN:
                    m_tuneDataPoints[i] = Mathf.Sin(t * 2.0f * Mathf.PI);
                    break;
                case INITTYPE.COS:
                    m_tuneDataPoints[i] = Mathf.Cos(t * 2.0f * Mathf.PI);
                    break;
                case INITTYPE.COS_INV_NORM:
                    m_tuneDataPoints[i] = (Mathf.Cos(t * 2.0f * Mathf.PI) - 1.0f) * -0.5f;
                    break;
                case INITTYPE.HALF_SIN:
                    m_tuneDataPoints[i] = Mathf.Sin(t * Mathf.PI);
                    break;
                default:
                    m_tuneDataPoints[i] = 0.5f;
                    break;
            }
        }

        // Update visualization
        while (m_curve.length > 0) m_curve.RemoveKey(0);
        for (int i = 0; i < s_size; i++)
        {
            float t = getTimeFromIdx(i);
            m_curve.AddKey(t,m_tuneDataPoints[i]);
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
        for (int i = 0; i < 100; i++)
        {
            float t = i * 0.01f;
            float t1 = (i+1) * 0.01f;
            Debug.DrawLine(transform.position + Vector3.right*t + Vector3.up * getValAt(t),
                transform.position + Vector3.right*t1 + Vector3.up * getValAt(t1),
                new Color((float)i/(float)99,0.5f,(float)i/(float)99));
        }
	}

    float getValAt(float p_t)
    {
        float realTime = (float)s_size * p_t;
        int low = Mathf.Min(s_size-1,Mathf.Max(0,(int)realTime));
        int hi = Mathf.Min(s_size-1,(int)(s_size*p_t + 1.0f));
        float lin = p_t * (float)s_size - (float)low;             
        //Debug.Log(p_t+": "+low + "->"+ hi+" [t"+lin+"]");
        //Debug.Log(hi);
        float val = 0.0f;
        try
        {
            val = Mathf.Lerp(m_tuneDataPoints[low], m_tuneDataPoints[hi], lin);
        }
        catch(Exception e)
        {
            Debug.Log(e.Message.ToString());
            Debug.Log(p_t + ": " + low + "->" + hi + " [t" + lin + "]");
        }
        return val;
    }


    void OnGUI()
    {
        m_curve = EditorGUILayout.CurveField("curve", m_curve);
    }
}
