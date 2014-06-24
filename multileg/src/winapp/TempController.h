#pragma once

// =======================================================================================
//                                      TempController
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Temporary controller class for camera, contains transform and
/// camera specific stuff
///        
/// # TempController
/// 
/// 28-9-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <MathHelp.h>
#include <math.h>
#include <utility>
using namespace std;

class TempController
{
public:
	TempController(float p_x, float p_y, float p_z, float p_yangleRad);
	virtual ~TempController();

	glm::mat4& getRotationMatrix();
	glm::mat4& getViewProjMatrix();
	glm::vec4& getPos();
	glm::quat& getRotation();
	float getAspect();

	void update(float p_dt);

	void setFovFromAngle(float angle, float aspectRatio);
	void setFovFromRad(float rad, float aspectRatio);

	void moveThrust(const glm::vec3& p_dir);
	void moveAngularThrust(const glm::vec3& p_dir);
	void rotate(glm::vec3 p_angularVelocity);

	float getVelocityAmount();

	bool isNewFovAvailable();
	glm::vec2& getFovXY();
protected:
private:
	void calcRotationMatrix();
	void calcViewProjMatrix(float p_fovYAngleDeg, float p_aspectRatio);

	glm::mat4 m_rotationMat;
	glm::mat4 m_viewProjMat;
	glm::mat4 m_projMat;

	glm::vec3 m_moveThrustDir;
	glm::vec3 m_moveAngularThrustDir;

	glm::vec4 m_velocity;
	glm::vec3 m_angularVelocity;

	float m_thrustPower;
	float m_angularThrustPower;
	float m_damping;
	float m_angularDamping;

	glm::vec4 m_position;
	glm::quat m_rotation;

	// camera stuff
	float m_fovYAngle, m_aspect;
	glm::vec2 m_fovTan;
	bool	m_fovDirtyBit;
};