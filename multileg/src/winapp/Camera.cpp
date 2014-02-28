/*#include "Camera.h"

Camera::Camera(float viewW, float viewH)
{
	 CalcFovFromAngle(70.0f); 
	 m_pos = glm::vec4(0.0f,0.0f,2.0f,1.0f); 
	 m_rot = glm::vec2(0.0f,0.0f);
	 m_rotMat = glm::mat4(1.0f);
	 m_lookAt = glm::vec4(0.0f,0.0f,-1.0f,1.0f);
	 m_fovDirtyBit=true;
}

void Camera::CalcFovFromAngle(float angle)
{
	CalcFovFromRad( angle*(float)TORAD );
}

void Camera::CalcFovFromRad(float rad)
{
	 m_fovx = rad;
	 m_fovy = m_viewHeight/m_viewWidth*m_fovx;
	 m_fovxTan=tan(m_fovx); 
	 m_fovyTan=tan(m_fovy);
	 m_fovDirtyBit=true;
}

// glm::mat4 Camera::GetViewMat()
// {
// 	glm::mat4 viewTranslate = glm::translate(glm::mat4(1.0f),m_pos);
// 	glm::mat4 viewRotateX = glm::rotate(viewTranslate,0.0f, glm::vec3(-1.0f, 0.0f, 0.0f));
// 	glm::mat4 view = glm::rotate(viewRotateX,10.0f, glm::vec3(0.0f, 1.0f, 0.0f)); 
// 	return view;
// }

void Camera::CalcViewMat(glm::vec3 offset)
{
// 	glm::vec3 pos = glm::vec3(m_pos.x,m_pos.y,m_pos.z);
// 	glm::vec3 lookAt = glm::vec3(m_lookAt.x,m_lookAt.y,m_lookAt.z);
// 	m_viewMat =  glm::lookAt(pos,
// 							pos+lookAt,
// 							glm::vec3(0.0f,1.0f,0.0f)
// 
// 							);
	glm::vec3 pos = glm::vec3(m_pos.x,m_pos.y,m_pos.z);
	
	
	glm::mat4 view = glm::lookAt(pos,
								pos + offset,
								glm::vec3(0.0f,1.0f,0.0f)
								);
	m_viewMat = view;

}

glm::mat4 Camera::GetViewMat()
{
	return m_viewMat;
}

void Camera::CalcRotMat()
{
	
// 	if (m_rot.x>TWOPI) m_rot.x-=TWOPI;
// 	if (m_rot.x<0.0f) m_rot.x+=TWOPI;
// 	if (m_rot.y>TWOPI) m_rot.y-=TWOPI;
// 	if (m_rot.y<0.0f) m_rot.y+=TWOPI;
// 	float degX = (float)(((m_rot.x))*TODEG);
// 	float degY = (float)(((m_rot.y))*TODEG);
// 	glm::mat4 rotX = glm::rotate(
// 		glm::mat4(1.0f),
// 		degX, glm::vec3(-1.0f, 0.0f, 0.0f));
// 	m_rotMat = glm::rotate(
// 		rotX,
// 		degY, glm::vec3(0.0f, 1.0f, 0.0f));
		
	// m_viewMat = m_rotMat;
}

glm::mat4 Camera::GetRotMat()
{
	return m_rotMat;
}

*/