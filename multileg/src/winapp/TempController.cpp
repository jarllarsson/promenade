#include "TempController.h"
#include <glm\gtc\type_ptr.hpp>
#include <ToString.h>
#include <DebugPrint.h>

TempController::TempController()
{
	m_position = glm::vec4(0.0f,0.0f,-10.0f,1.0f);
	m_rotation = glm::quat(glm::vec3(0.0f,0.2f,0.0f));
	m_fovDirtyBit = false;

	m_damping=2.0f;
	m_angularDamping=1.5f;
	m_thrustPower=20.0f;
	m_angularThrustPower=2.0f;
}

TempController::~TempController()
{

}

void TempController::setFovFromAngle( float angle, float aspectRatio )
{
	setFovFromRad( angle*(float)TORAD, aspectRatio );
}

void TempController::setFovFromRad( float rad, float aspectRatio )
{
	float fovxRad = rad*0.5f;
	float fovyRad = fovxRad;
	m_fovTan.x=aspectRatio*tan(fovxRad); 
	m_fovTan.y=tan(fovyRad);
	m_fovDirtyBit=true;
}


bool TempController::isNewFovAvailable()
{
	return m_fovDirtyBit;
}

glm::vec2& TempController::getFovXY()
{
	 m_fovDirtyBit=false; 
	 return m_fovTan;
}

glm::mat4& TempController::getRotationMatrix()
{
	return m_rotationMat;
}


glm::vec4& TempController::getPos()
{
	return m_position;
}

void TempController::update( float p_dt )
{
	// normalize input
	glm::normalize(m_moveThrustDir);
	//glm::normalize(m_moveAngularThrustDir);

	// apply damping
	m_velocity -= m_velocity*m_damping*p_dt;
	m_angularVelocity -= m_angularVelocity*m_angularDamping*p_dt;

	// apply "force" vector on rotation
	m_angularVelocity += m_moveAngularThrustDir*m_angularThrustPower*p_dt;
	

	// update rotation
	rotate(m_angularVelocity*p_dt);

	// calc new rotation
	calcRotationMatrix();

	// apply "force" vector on velocity
	m_velocity += glm::vec4(m_moveThrustDir*m_thrustPower*p_dt,0.0f)*m_rotationMat;

	// update position
	m_position += m_velocity*p_dt;

	// restore input
	m_moveThrustDir=glm::vec3();
	m_moveAngularThrustDir=glm::vec3();
}

void TempController::moveThrust(const glm::vec3& p_dir)
{
	m_moveThrustDir+=p_dir;
}

void TempController::moveAngularThrust( const glm::vec3& p_dir)
{
	m_moveAngularThrustDir+=p_dir;
}

void TempController::calcRotationMatrix()
{
	m_rotationMat=glm::toMat4(m_rotation);
}

void TempController::rotate( glm::vec3 p_angularVelocity )
{
	if (glm::sqrLength(p_angularVelocity)>0.0f)
	{
		glm::quat turn = glm::quat(p_angularVelocity);
		m_rotation = turn*m_rotation;
	}
}
