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
	TempController();
	virtual ~TempController();

	glm::mat4& getRotationMatrix();
	glm::vec4& getPos();

	void update(float p_dt);

	void setFovFromAngle(float angle, float aspectRatio);
	void setFovFromRad(float rad, float aspectRatio);

	void moveThrust(const glm::vec3& p_dir);
	void moveAngularThrust(const glm::vec3& p_dir);
	void rotate(glm::vec3 p_angularVelocity);

	bool isNewFovAvailable();
	glm::vec2& getFovXY();
protected:
private:
	void calcRotationMatrix();

	glm::mat4 m_rotationMat;

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
	glm::vec2 m_fovTan;
	bool	m_fovDirtyBit;
};