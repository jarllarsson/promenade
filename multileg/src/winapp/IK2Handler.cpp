#include "IK2Handler.h"
#include <MathHelp.h>
#include "DebugDrawBatch.h"
#include <algorithm> 


IK2Handler::IK2Handler()
{
	float m_upperAngle=0.0f;
	float m_lowerAngle=0.0f;
}

IK2Handler::~IK2Handler()
{

}


void IK2Handler::solve(const glm::vec3& p_footPos, const glm::vec3& p_upperLegJointPos, float p_upperLegLen, float p_lowerLegLen, DebugDrawBatch* p_drawer)
{
	int kneeFlip = 1;
	// Retrieve the current wanted foot position
	glm::vec3 footPos = p_footPos;
	glm::vec3 upperLegPos = p_upperLegJointPos;

	// Vector between foot and hip
	glm::vec3 topToFoot = footPos-upperLegPos;
	//topToFoot.z *= -1;
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
		float cosVal = (hBS + uBS - lBS) / std::max(0.001f, (2.0f * uB * toFootLen));
		cosVal = std::min(std::max(cosVal, -1.0f), 1.0f);
// 		if (cosVal < -1.0f)
// 			int x = 0;
 		float acosVal = acos(cosVal);
// 		if (std::isnan(acosVal))
// 			int i = 0;
		upperLegAngle = (float)(kneeFlip)* acosVal + offsetAngle;
		// second angle
		glm::vec2 newLeg(uB * std::cos(upperLegAngle), uB * std::sin(upperLegAngle));
		lowerLegAngle = (MathHelp::satan2(topToFoot.y - newLeg.y, topToFoot.z - newLeg.x) - upperLegAngle);
	}
	else // otherwise, straight leg
	{
		upperLegAngle = offsetAngle;
		lowerLegAngle = 0.0f;
	}
	updateAngles(lowerLegAngle, upperLegAngle);
	m_lowerAngleW = upperLegAngle + lowerLegAngle;


	m_upperAngle = upperLegAngle;

	// Debug draw bones
	m_kneePos = glm::vec3(0.0f, uB * std::sin(upperLegAngle), uB * std::cos(upperLegAngle));
	m_endPos = glm::vec3(0.0f, lB * std::sin(m_lowerAngleW), lB * std::cos(m_lowerAngleW));
	//
	m_kneePos += upperLegPos;
	m_endPos += m_kneePos;
	m_hipPos = p_upperLegJointPos;

	if (p_drawer!=NULL) debugDraw(p_drawer);
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
	return m_lowerAngleW;
}

void IK2Handler::updateAngles(float p_lowerAngle, float p_upperAngle)
{
	m_lowerAngle = p_lowerAngle;
	m_upperAngle = p_upperAngle;
}

glm::vec3& IK2Handler::getHipPos()
{
	return m_hipPos;
}

glm::vec3& IK2Handler::getKneePos()
{
	return m_kneePos;
}

glm::vec3& IK2Handler::getFootPos()
{
	return m_endPos;
}

void IK2Handler::debugDraw(DebugDrawBatch* p_drawer)
{
	p_drawer->drawLine(m_hipPos, m_kneePos, dawnBringerPalRGB[COL_PURPLE], dawnBringerPalRGB[COL_PINK]);
	p_drawer->drawLine(m_kneePos, m_endPos, dawnBringerPalRGB[COL_PINK], dawnBringerPalRGB[COL_SLIMEGREEN]);
	p_drawer->drawLine(m_endPos + 0.1f*glm::vec3(-1, 0, 1), m_endPos +	 0.1f*glm::vec3(1, 0, 1), dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
	p_drawer->drawLine(m_endPos + 0.1f*glm::vec3(1, 0, 1), m_endPos +	 0.1f*glm::vec3(1, 0, -1), dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
	p_drawer->drawLine(m_endPos + 0.1f*glm::vec3(1, 0, -1), m_endPos +	 0.1f*glm::vec3(-1, 0, -1), dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
	p_drawer->drawLine(m_endPos + 0.1f*glm::vec3(-1, 0, -1), m_endPos +  0.1f*glm::vec3(-1, 0, 1), dawnBringerPalRGB[COL_SLIMEGREEN], dawnBringerPalRGB[COL_SLIMEGREEN]);
}
