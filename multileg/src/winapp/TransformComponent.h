#pragma once

#include <Artemis.h>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <MathHelp.h>
// =======================================================================================
//                                      TransformComponent
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # TransformComponent
/// 
/// 15-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------


class TransformComponent : public artemis::Component
{

public:
	TransformComponent(float p_x = 0.0f, float p_y = 0.0f, float p_z = 0.0f)
	{
		setPositionToMatrix(glm::vec3(p_x,p_y,p_z));
		setRotationToMatrix(glm::quat(0.0f, 0.0f, 0.0f, 1.0f));
		setScaleToMatrix(glm::vec3(1.0f));
		updateMatrix();
	};

	TransformComponent(glm::vec3& p_position = glm::vec3(0.0f), 
					   glm::quat& p_rotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f),
					   glm::vec3& p_scale  = glm::vec3(1.0f))
	{
		setPositionToMatrix(p_position);
		setRotationToMatrix(p_rotation);
		setScaleToMatrix(p_scale);
		updateMatrix();
	};


	void setPositionToMatrix(glm::vec3& p_position);
	void setRotationToMatrix(glm::quat& p_rotation);
	void setScaleToMatrix(glm::vec3& p_scale);
	void setMatrixToComponents(glm::mat4& p_mat);

	glm::vec3&	getPosition() { return m_position; }
	glm::quat&	getRotation() { return m_rotation; }
	glm::vec3&	getScale() { return m_scale; }
	glm::mat4&	getMatrix() { return m_transform; }

	void updateMatrix();

private:
	glm::vec3 m_position;
	glm::quat m_rotation;
	glm::vec3 m_scale;
	glm::mat4 m_transform;
};

void TransformComponent::setPositionToMatrix(glm::vec3& p_position)
{
	m_position = p_position;
	updateMatrix();
}

void TransformComponent::setRotationToMatrix(glm::quat& p_rotation)
{
	m_rotation = p_rotation;
	updateMatrix();
}

void TransformComponent::setScaleToMatrix(glm::vec3& p_scale)
{
	m_scale = p_scale;
	updateMatrix();
}

void TransformComponent::setMatrixToComponents(glm::mat4& p_mat)
{
	m_transform = p_mat;
	glm::mat4 rotation;
	// decompose the matrix to its components
	MathHelp::decomposeTRS(m_transform, m_scale, rotation, m_position);
	// get rotation quat
	m_rotation = glm::quat(rotation);
}

// Update transform from component values
void TransformComponent::updateMatrix()
{
	glm::mat4 translate = glm::translate(glm::mat4(1.0f), m_position);
	glm::mat4 rotate = glm::mat4_cast(m_rotation);
	// decomposeTRS for scale
	glm::mat4 scale = glm::scale(glm::mat4(1.0f), m_scale);
	m_transform = translate * rotate * scale; // is the scale * rotate * translate
}
