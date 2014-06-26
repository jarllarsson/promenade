#pragma once
#include <glm\gtc\type_ptr.hpp>
#include <ColorPalettes.h>
#include <vector>
// =======================================================================================
//                                      DebugDrawBatch
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # DebugDrawBatch
/// 
/// 25-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class DebugDrawBatch
{
public:
	// Line primitive
	struct Line
	{
		glm::vec3 m_start;
		glm::vec3 m_end;
		Color4f m_startColor, m_endColor;
	};

	DebugDrawBatch() {}
	virtual ~DebugDrawBatch() {}

	virtual void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color4f& p_color);

	virtual void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color3f& p_color);

	virtual void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color3f& p_startColor, const Color3f& p_endColor);

	virtual void drawLine(const glm::vec3& p_start, const glm::vec3& p_end,
		const Color4f& p_startColor, const Color4f& p_endColor);


	void clearDrawCalls();

	std::vector<Line>* getLineList();
	void clearLineList();

protected:
private:

	std::vector<Line> m_lineList;
};
