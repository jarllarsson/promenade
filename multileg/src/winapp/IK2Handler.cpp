#include "IK2Handler.h"
#include "..\Util\MathHelp.h"


IK2Handler::IK2Handler()
{

}

IK2Handler::~IK2Handler()
{

}


void IK2Handler::solve( const glm::vec3& p_footPosL, const glm::vec3& p_upperLegJointPosL, float p_upperLegLen, float p_lowerLegLen )
{
	int kneeFlip = 1;
	// Retrieve the current wanted foot position
	glm::vec3 footPos = p_footPosL;
	glm::vec3 upperLegPos = p_upperLegJointPosL;

	// Vector between foot and hip
	glm::vec3 topToFoot = upperLegPos - footPos;
	topToFoot.y *= -1;


	// This ik calc is in 2d, so eliminate rotation
	// Use the coordinate system of the leg frame as
	// in the paper
	topToFoot.x = 0.0f; // squish x axis
	//
	float toFootLen = glm::length(topToFoot);
	float upperLegAngle = 0.0f;
	float lowerLegAngle = 0.0f;
	float uB = p_upperLegLen; // the length of the legs
	float lB = p_lowerLegLen;
	// first get offset angle beetween foot and axis
	float offsetAngle = MathHelp::satan2(topToFoot.y, topToFoot.z);
	// If dist to foot is shorter than combined leg length
	if (toFootLen < uB + lB)
	{
		float uBS = uB * uB;
		float lBS = lB * lB;
		float hBS = toFootLen * toFootLen;
		// law of cosines for first angle
		upperLegAngle = (float)(kneeFlip)* acos((hBS + uBS - lBS) / (2.0f * uB * toFootLen)) + offsetAngle;
		// second angle
		glm::vec2 newLeg(uB * cos(upperLegAngle), uB * sin(upperLegAngle));
		lowerLegAngle = MathHelp::satan2(topToFoot.y - newLeg.y, topToFoot.z - newLeg.x) - upperLegAngle;
	}
	else // otherwise, straight leg
	{
		upperLegAngle = offsetAngle;
		lowerLegAngle = 0.0f;
	}
	float lowerAngleW = upperLegAngle + lowerLegAngle;


	m_hipAngle = upperLegAngle;
	m_kneeAngle = lowerAngleW;

	// Debug draw bones
	m_kneePos = glm::vec3(0.0f, uB * sin(upperLegAngle), uB * cos(upperLegAngle));
	m_endPos = glm::vec3(0.0f, lB * sin(lowerAngleW), lB * cos(lowerAngleW));
	//Debug.Log("upper angle: " + upperLegAngle);
	if (m_legFrame != null)
	{
		m_kneePos = upperLegPos + m_kneePos/* m_legFrame.transform.TransformDirection(m_kneePos)*/;
		m_endPos = m_kneePos + m_endPos/*m_legFrame.transform.TransformDirection()*/;

		// PID test
		//if (m_testPIDLower != null && m_testPIDUpper != null)
		//{
		//	Quaternion localUpper = /*Quaternion.Inverse(m_legFrame.transform.rotation) **/ m_upperLeg.rotation;
		//	Quaternion localLower = Quaternion.Inverse(m_upperLeg.rotation) * m_lowerLeg.rotation;
		//	Quaternion localGoalUpper = Quaternion.AngleAxis(Mathf.Rad2Deg * (upperLegAngle + Mathf.PI*0.5f), Vector3.left);
		//	Quaternion localGoalLower = Quaternion.AngleAxis(Mathf.Rad2Deg * (lowerLegAngle/* - Mathf.PI*0.5f*/), Vector3.left);
		//	m_testPIDUpper.drive(localUpper, localGoalUpper, Time.deltaTime);
		//	m_testPIDLower.drive(localLower, localGoalLower, Time.deltaTime);
		//}
	}
	else
	{
		m_kneePos += upperLegPos;
		m_endPos += m_kneePos;
	}

	//glm::vec3 offset;
	//int idx = (int)m_legType;
	//if (m_legFrame != null)
	//	offset = new Vector3(m_legFrame.m_footTarget[idx].x, 0.0f/*m_legFrame.transform.position.y*/, m_startPos.z + m_legFrame.getOptimalProgress() /*-m_legFrame.getReferenceLiftPos(idx).z/* + m_legFrame.transform.position*/);
	////offset = new Vector3(m_legFrame.getReferenceFootPos(idx).x, m_legFrame.transform.position.y, m_legFrame.getReferenceFootPos(idx).z);
	//else if (m_foot != null)
	//	offset = new Vector3(m_upperLeg.position.x, 0.0f, m_upperLeg.position.z)/* + m_legFrame.transform.position*/;

	//m_hipPos = offset + upperLegPos;
	//m_kneePosW = offset + m_kneePos;
	//Debug.DrawLine(offset + upperLegLocalPos, m_kneePosW, Color.red);
	//Debug.DrawLine(m_kneePosW, offset + m_endPos, Color.blue);


	//if (m_dbgMesh)
	//{
	//	m_dbgMesh.rotation = m_legFrame.transform.rotation * Quaternion.AngleAxis(Mathf.Rad2Deg * (upperLegAngle + Mathf.PI*0.5f), -m_legFrame.transform.right);
	//	m_dbgMesh.position = upperLegLocalPos;
	//}
}
