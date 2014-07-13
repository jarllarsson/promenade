#include "IK2Handler.h"
#include "..\Util\MathHelp.h"
#include "DebugDrawBatch.h"


IK2Handler::IK2Handler()
{
	float m_upperAngle=0.0f;
	float m_lowerAngle=0.0f;
}

IK2Handler::~IK2Handler()
{

}


void IK2Handler::solve(const glm::vec3& p_footPosL, const glm::vec3& p_upperLegJointPosL, float p_upperLegLen, float p_lowerLegLen, DebugDrawBatch* p_drawer)
{
	int kneeFlip = 1;
	// Retrieve the current wanted foot position
	glm::vec3 footPos = p_footPosL;
	glm::vec3 upperLegPos = p_upperLegJointPosL;

	// Vector between foot and hip
	glm::vec3 topToFoot = footPos-upperLegPos;
	//topToFoot.y *= -1;


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
	updateAngles(lowerLegAngle, upperLegAngle);
	float lowerAngleW = upperLegAngle + lowerLegAngle;


	m_upperAngle = upperLegAngle;
	m_lowerAngle = lowerAngleW;

	// Debug draw bones
	m_kneePos = glm::vec3(0.0f, uB * sin(upperLegAngle), uB * cos(upperLegAngle));
	m_endPos = glm::vec3(0.0f, lB * sin(lowerAngleW), lB * cos(lowerAngleW));
	//
	m_kneePos += upperLegPos;
	m_endPos += m_kneePos;
	m_hipPos = p_upperLegJointPosL;


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
	debugDraw(p_drawer);
}

float IK2Handler::getUpperLegAngle() const
{
	return m_upperAngle;
}

float IK2Handler::getLowerLocalLegAngle() const
{
	return m_lowerAngle;
}

float IK2Handler::getLowerWorldLegAngle() const
{
	return m_upperAngle + m_lowerAngle;
}

void IK2Handler::updateAngles(float p_lowerAngle, float p_upperAngle)
{
	m_lowerAngle = p_lowerAngle;
	m_upperAngle = p_upperAngle;
}

glm::vec3& IK2Handler::getHipPosL()
{
	return m_hipPos;
}

glm::vec3& IK2Handler::getKneePosL()
{
	return m_kneePos;
}

glm::vec3& IK2Handler::getFootPosL()
{
	return m_endPos;
}

void IK2Handler::debugDraw(DebugDrawBatch* p_drawer)
{
	p_drawer->drawLine(m_hipPos, m_kneePos, dawnBringerPalRGB[COL_PURPLE], dawnBringerPalRGB[COL_PINK]);
	p_drawer->drawLine(m_kneePos, m_endPos, dawnBringerPalRGB[COL_PINK], dawnBringerPalRGB[COL_SLIMEGREEN]);
	p_drawer->drawLine(m_endPos + glm::vec3(-1, 0, 1), m_endPos + glm::vec3(1, 0, 1), dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
	p_drawer->drawLine(m_endPos + glm::vec3(1, 0, 1), m_endPos + glm::vec3(1, 0, -1), dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
	p_drawer->drawLine(m_endPos + glm::vec3(1, 0, -1), m_endPos + glm::vec3(-1, 0, -1), dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
	p_drawer->drawLine(m_endPos + glm::vec3(-1, 0, -1), m_endPos + glm::vec3(-1, 0, 1), dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
}
