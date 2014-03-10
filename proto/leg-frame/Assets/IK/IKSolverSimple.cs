﻿using UnityEngine;
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
            footPos = m_legFrame.m_footTarget[(int)m_legType];

        // Vector between foot and hip
        Vector3 topToFoot = footPos - m_upperLeg.position;
        Debug.DrawLine(m_upperLeg.position, m_upperLeg.position+topToFoot,Color.black);
        // This ik calc is in 2d, so eliminate rotation
        // Use the coordinate system of the leg frame as
        // in the paper
        if (m_legFrame != null)
            topToFoot = m_legFrame.transform.InverseTransformDirection(topToFoot);
        //Debug.DrawLine(m_upperLeg.position, m_upperLeg.position + topToFoot, Color.yellow);
        topToFoot.x = 0.0f; // squish x axis
        //Debug.DrawLine(m_upperLeg.position, m_upperLeg.position + topToFoot, Color.yellow*2.0f);
        //
        float toFootLen = topToFoot.magnitude;
        float upperLegAngle = 0.0f;
        float lowerLegAngle = 0.0f;
        float uB = m_upperLeg.localScale.y; // the length of the legs
        float lB = m_lowerLeg.localScale.y;
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

            Vector3 kneePosT = m_upperLeg.position + new Vector3(0.0f, newLeg.y, newLeg.x);
            //Debug.DrawLine(m_upperLeg.position, kneePosT,Color.magenta);

            lowerLegAngle = Mathf.Atan2(topToFoot.y - newLeg.y, topToFoot.z - newLeg.x) - upperLegAngle;
            /*lowerLegAngle = acos((uBS + lBS - hBS)/(2.0f*uB*lB))-upperLegAngle;*/
        }
        else // otherwise, straight leg
        {
            upperLegAngle = offsetAngle;

            Vector2 newLeg = new Vector2(uB * Mathf.Cos(upperLegAngle), uB * Mathf.Sin(upperLegAngle));

            Vector3 kneePosT = m_upperLeg.position + new Vector3(0.0f, newLeg.y, newLeg.x);
            //Debug.DrawLine(m_upperLeg.position, kneePosT, Color.magenta);

            lowerLegAngle = 0.0f;
            //straightLeg = true;
        }
        float lowerAngleW = upperLegAngle + lowerLegAngle;


        m_hipAngle = upperLegAngle;
        m_kneeAngle = lowerAngleW;

        // Debug draw bones
        Vector3 kneePos=new Vector3(0.0f, uB *Mathf.Sin(upperLegAngle), uB *Mathf.Cos(upperLegAngle));        
        Vector3 endPos = new Vector3(0.0f, lB * Mathf.Sin(lowerAngleW), lB * Mathf.Cos(lowerAngleW));
        if (m_legFrame != null)
        {
            kneePos = m_upperLeg.position + m_legFrame.transform.TransformDirection(kneePos);
            endPos = kneePos + m_legFrame.transform.TransformDirection(endPos);
            // PID test
            Quaternion goal = m_legFrame.transform.rotation * Quaternion.AngleAxis(Mathf.Rad2Deg * (upperLegAngle + Mathf.PI*0.5f), -m_legFrame.transform.right);
            m_testPIDUpper.drive(m_upperLeg.rotation, goal, Time.deltaTime);
        }
        else
        {
            kneePos += m_upperLeg.position;
            endPos += kneePos;
        }

        Debug.DrawLine(m_upperLeg.position, kneePos);
        Debug.DrawLine(kneePos, endPos);

        if (m_dbgMesh)
        {
            m_dbgMesh.rotation = m_legFrame.transform.rotation * Quaternion.AngleAxis(Mathf.Rad2Deg * (upperLegAngle + Mathf.PI*0.5f), -m_legFrame.transform.right);
            m_dbgMesh.position = m_upperLeg.position;
        }
    }
}
