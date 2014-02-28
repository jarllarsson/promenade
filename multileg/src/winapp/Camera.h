#pragma once


/*#include <glm/glm.hpp>
#include "glm/gtx/quaternion.hpp"
#include <glm/gtc/matrix_transform.hpp> 
#include <MathHelp.h>
#include <math.h>
#include <utility>
using namespace std;

class Camera 
{
public:
	Camera(float viewW, float viewH);

	void CalcFovFromAngle(float angle);
	void CalcFovFromRad(float rad);

	void CalcRotMat();
	glm::mat4 GetRotMat();

	void CalcViewMat(glm::vec3 offset);
	glm::mat4 GetViewMat();

	bool IsNewFovAvailable() {return m_fovDirtyBit;}

	glm::vec2 GetFinalFovXY() {m_fovDirtyBit=false; return glm::vec2(m_fovxTan,m_fovyTan);}
	glm::vec4 m_pos;
	glm::vec4 m_lookAt;
	glm::vec2 m_rot;
private:	
	float m_viewWidth, m_viewHeight;
	float m_fovx;		
	float m_fovy;
	float m_fovxTan;
	float m_fovyTan;
	bool m_fovDirtyBit;
	glm::mat4 m_rotMat;
	glm::mat4 m_viewMat;
};
*/