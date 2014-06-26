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

void DebugDrawBatch::clearLineList()
{
	m_lineList.clear();
}

std::vector<DebugDrawBatch::Line>* DebugDrawBatch::getLineList()
{
	return &m_lineList;
}

void DebugDrawBatch::clearDrawCalls()
{
	clearLineList();
}
