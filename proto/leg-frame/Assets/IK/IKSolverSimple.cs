using UnityEngine;
using System.Collections;

public class IKSolverSimple : MonoBehaviour 
{
    public LegFrame.LEG m_legType;
    public Transform m_upperLeg;
    public Transform m_lowerLeg;
    public Transform m_foot;
    public Transform m_dbgMesh;
    public LegFrame m_legFrame;
    public float m_hipAngle;
    public float m_kneeAngle;
    public PIDn m_testPIDUpper;
    public PIDn m_testPIDLower;
    public Vector3 m_kneePos;
    public Vector3 m_endPos;

	// Use this for initialization
	void Start () 
    {
	
	}
	
	// Update is called once per frame
	void Update () 
    {
        calculate();
	}

    void calculate()
    {
        int kneeFlip = 1;
        // Retrieve the current wanted foot position
        Vector3 footPos;
        if (m_foot!=null)
            footPos = m_foot.position;
        else
        {
            // get non-corrected foot pos here
            footPos = new Vector3(0.0f, m_legFrame.getGraphedFootPos((int)m_legType), 0.0f/*m_legFrame.m_footTarget[(int)m_legType].z-m_legFrame.transform.position.z*/);
        }
            //footPos = m_legFrame.m_footTarget[(int)m_legType];
     
        //footPos = new Vector3(footPos.x, 0.0f, footPos.z);        
        //footPos -= m_legFrame.transform.position;   
        Vector3 upperLegLocalPos = (m_upperLeg.position - m_legFrame.transform.position);
        upperLegLocalPos = new Vector3(0.0f, m_upperLeg.localScale.y + m_lowerLeg.localScale.y + 0.2f, 0.0f);

        Debug.DrawLine(footPos, footPos + Vector3.up, Color.black);

        // Vector between foot and hip
        Vector3 topToFoot = upperLegLocalPos - footPos;
        
        //Debug.DrawLine(m_upperLeg.position, m_upperLeg.position+topToFoot,Color.black);

        // This ik calc is in 2d, so eliminate rotation
        // Use the coordinate system of the leg frame as
        // in the paper
        /*if (m_legFrame != null)
            topToFoot = m_legFrame.transform.InverseTransformDirection(topToFoot);*/
        //Debug.DrawLine(m_upperLeg.position, m_upperLeg.position + topToFoot, Color.yellow);
        topToFoot.x = 0.0f; // squish x axis
        //Debug.DrawLine(m_upperLeg.position, m_upperLeg.position + topToFoot, Color.yellow*2.0f);
        //
        float toFootLen = topToFoot.magnitude*1.0f;
        float upperLegAngle = 0.0f;
        float lowerLegAngle = 0.0f;
        float uB = m_upperLeg.localScale.y*1.0f; // the length of the legs
        float lB = m_lowerLeg.localScale.y*1.0f;
        //Debug.Log(uB);
        // first get offset angle beetween foot and axis
        float offsetAngle = Mathf.Atan2(topToFoot.y, topToFoot.z);
        // If dist to foot is shorter than combined leg length
        //bool straightLeg = false;
        if (toFootLen < uB + lB)
        {
            float uBS = uB * uB;
            float lBS = lB * lB;
            float hBS = toFootLen * toFootLen;
            // law of cosines for first angle
            upperLegAngle = (float)(kneeFlip) * Mathf.Acos((hBS + uBS - lBS) / (2.0f * uB * toFootLen)) + offsetAngle;
            // second angle
            Vector2 newLeg = new Vector2(uB * Mathf.Cos(upperLegAngle), uB * Mathf.Sin(upperLegAngle));

            Vector3 kneePosT = upperLegLocalPos + new Vector3(0.0f, newLeg.y, newLeg.x);
            //Debug.DrawLine(m_upperLeg.position, kneePosT,Color.magenta);

            lowerLegAngle = Mathf.Atan2(topToFoot.y - newLeg.y, topToFoot.z - newLeg.x) - upperLegAngle;
            /*lowerLegAngle = acos((uBS + lBS - hBS)/(2.0f*uB*lB))-upperLegAngle;*/
        }
        else // otherwise, straight leg
        {
            upperLegAngle = offsetAngle;

            Vector2 newLeg = new Vector2(uB * Mathf.Cos(upperLegAngle), uB * Mathf.Sin(upperLegAngle));

            Vector3 kneePosT = upperLegLocalPos + new Vector3(0.0f, newLeg.y, newLeg.x);
            //Debug.DrawLine(m_upperLeg.position, kneePosT, Color.magenta);

            lowerLegAngle = 0.0f;
            //straightLeg = true;
        }
        float lowerAngleW = upperLegAngle + lowerLegAngle;


        m_hipAngle = upperLegAngle;
        m_kneeAngle = lowerAngleW;

        // Debug draw bones
        m_kneePos = new Vector3(0.0f, uB * Mathf.Sin(upperLegAngle), uB * Mathf.Cos(upperLegAngle));
        m_endPos = new Vector3(0.0f, lB * Mathf.Sin(lowerAngleW), lB * Mathf.Cos(lowerAngleW));
        if (m_legFrame != null)
        {
            m_kneePos = upperLegLocalPos + m_kneePos/* m_legFrame.transform.TransformDirection(m_kneePos)*/;
            m_endPos = m_kneePos + m_endPos/*m_legFrame.transform.TransformDirection()*/;
            // PID test
            Quaternion localUpper = Quaternion.Inverse(m_legFrame.transform.rotation) * m_upperLeg.rotation;
            Quaternion localLower = Quaternion.Inverse(m_upperLeg.rotation) * m_lowerLeg.rotation;
            Quaternion localGoalUpper = Quaternion.AngleAxis(Mathf.Rad2Deg * (upperLegAngle /*+ Mathf.PI*0.5f*/), Vector3.left);
            Quaternion localGoalLower = Quaternion.AngleAxis(Mathf.Rad2Deg * (lowerLegAngle/* + Mathf.PI*0.5f*/), Vector3.left);
            m_testPIDUpper.drive(localUpper, localGoalUpper, Time.deltaTime);
            m_testPIDLower.drive(localLower, localGoalLower, Time.deltaTime);
        }
        else
        {
            m_kneePos += upperLegLocalPos;
            m_endPos += m_kneePos;
        }

        Vector3 offset = new Vector3(m_legFrame.m_footTarget[(int)m_legType].x, m_legFrame.transform.position.y, m_legFrame.m_footTarget[(int)m_legType].z)/* + m_legFrame.transform.position*/;
        
        Debug.DrawLine(offset + m_kneePos, offset + m_endPos, Color.blue);
        Debug.DrawLine(offset + upperLegLocalPos, offset + m_kneePos,Color.red);

        if (m_dbgMesh)
        {
            m_dbgMesh.rotation = m_legFrame.transform.rotation * Quaternion.AngleAxis(Mathf.Rad2Deg * (upperLegAngle + Mathf.PI*0.5f), -m_legFrame.transform.right);
            m_dbgMesh.position = upperLegLocalPos;
        }
    }
}
