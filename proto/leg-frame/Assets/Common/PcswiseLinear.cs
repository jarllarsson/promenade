using UnityEngine;
using System.Collections;
using UnityEditor;

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
	void Update () {
	
	}

    float getValAt(float p_t)
    {
        float low = (float)((int)p_t);
        float hi = (float)((int)(p_t + 0.5f));
        float lin = p_t - low;
        return Mathf.Lerp(low, hi, lin);
    }

    void OnGUI()
    {
        m_curve = EditorGUILayout.CurveField("curve", m_curve);
    }
}
