#include "DebugDrawBatch.h"



void DebugDrawBatch::drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
	const Color4f& p_color)
{
	drawLine(p_start, p_end, p_color, p_color);
}
void DebugDrawBatch::drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
	const Color3f& p_color)
{
	drawLine(p_start, p_end, toColor4f(p_color), toColor4f(p_color));
}
void DebugDrawBatch::drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
	const Color3f& p_startColor, const Color3f& p_endColor)
{
	drawLine(p_start, p_end, toColor4f(p_startColor), toColor4f(p_endColor));
}
void DebugDrawBatch::drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
	const Color4f& p_startColor, const Color4f& p_endColor)
{
	Line line = { p_start, p_end, p_startColor, p_endColor };
	m_lineList.push_back(line);
}



void DebugDrawBatch::drawSphere(const glm::vec3& p_pos, float p_rad, const Color4f& p_color)
{
	Sphere sphere = { p_pos, p_rad, p_color};
	m_lineList.push_back(sphere);
}

void DebugDrawBatch::drawSphere(const glm::vec3& p_pos, float p_rad, const Color3f& p_color)
{
	Sphere sphere = { p_pos, p_rad, toColor4f(p_color) };
	m_sphereList.push_back(sphere);
}

void DebugDrawBatch::clearLineList()
{
	m_lineList.clear();
}



void DebugDrawBatch::clearSphereList()
{
	m_sphereList.clear();
}

std::vector<DebugDrawBatch::Line>* DebugDrawBatch::getLineList()
{
	return &m_lineList;
}

std::vector<DebugDrawBatch::Sphere>* DebugDrawBatch::getSphereList()
{
	return &m_sphereList;
}

void DebugDrawBatch::clearDrawCalls()
{
	clearLineList();
	clearSphereList();
}

