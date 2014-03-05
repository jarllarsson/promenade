using UnityEngine;
using System.Collections;

public class PIDvisualizer : MonoBehaviour 
{
    public PID m_pid;

    public float m_timelineLen = 5.0f;
    private float m_currTime = 0.0f;
    private float m_oldErr;
    public float m_divider=1.0f;
    public float m_scale = 1.0f;
    private bool m_inited = false;

	// Use this for initialization
	void Start () 
    {

	}
	
	// Update is called once per frame
	void LateUpdate () 
    {
        if (m_divider < Mathf.Abs(m_pid.m_P))
            m_divider = Mathf.Abs(m_pid.m_P);
        // Store old time
        float oldTime=m_currTime;
        // move time forward
        m_currTime+=Time.deltaTime;
        if (m_currTime>=m_timelineLen || m_currTime<0.0f)
        {
            m_currTime=0.0f;
            oldTime=m_currTime;
            //m_divider = 1.0f;
        }
        if (oldTime > m_currTime) oldTime = m_currTime;
        // Draw timeline
        Debug.DrawLine(transform.position - m_scale * Vector3.right, transform.position + m_scale * Vector3.right, Color.black);
        // Get current error
        float err = m_pid.m_P;
        if (!m_inited) { m_oldErr = err; m_inited = true; }
        // Draw graph and let it be visible for the remaining timeline time
        float nt = m_currTime / m_timelineLen; // normalized timeline pos
        float ont = oldTime / m_timelineLen; // normalized old timeline pos
        // Draw current 
        Vector3 origo = transform.position - m_scale*Vector3.right;
        Vector3 start = origo + m_scale * 2.0f * Vector3.right * ont + m_scale * Vector3.up * m_oldErr / m_divider;
        Vector3 end = origo + m_scale * 2.0f * Vector3.right * nt + m_scale * Vector3.up * err / m_divider;
        float t = (1.0f - (Mathf.Abs(err) / m_divider));
        Color c=Color.Lerp(Color.red,Color.green,t*t);
        Debug.DrawLine(start, end, c,m_timelineLen-m_currTime);
        // also draw onion skin x2
        Color ca = new Color(c.r, c.g, c.b, 0.3f);
        Debug.DrawLine(start, end, ca, m_timelineLen+(m_timelineLen - m_currTime));
        ca = new Color(c.r, c.g, c.b, 0.1f);
        Debug.DrawLine(start, end, ca, m_timelineLen*2.0f + (m_timelineLen - m_currTime));

        // store error
        m_oldErr = err;      

	}
}
