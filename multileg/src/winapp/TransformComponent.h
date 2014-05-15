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
		m_matrixDirty = false;
	};

	TransformComponent(glm::vec3& p_position = glm::vec3(0.0f), 
					   glm::quat& p_rotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f),
					   glm::vec3& p_scale  = glm::vec3(1.0f))
	{
		setPositionToMatrix(p_position);
		setRotationToMatrix(p_rotation);
		setScaleToMatrix(p_scale);
		updateMatrix();
		m_matrixDirty = false;
	};


	void setPositionToMatrix(const glm::vec3& p_position);
	void setRotationToMatrix(const glm::quat& p_rotation);
	void setScaleToMatrix(const glm::vec3& p_scale);

	void setPosRotToMatrix(const glm::vec3& p_position, const glm::quat& p_rotation);
	void setPosScaleToMatrix(const glm::vec3& p_position, const glm::vec3& p_scale);
	void setRotScaleToMatrix(const glm::quat& p_rotation, const glm::vec3& p_scale);
	void setAllToMatrix(const glm::vec3& p_position, const glm::quat& p_rotation, const glm::vec3& p_scale);

	void setMatrixToComponents(const glm::mat4& p_mat);

	// Allow for direct handling of matrices
	// this call will set the matrix as dirty
	// matrix decomposition will only occur when a component is accessed
	void setMatrix(const glm::mat4& p_mat);

	const glm::vec3& getPosition() { updateOnDirty(); return m_position; }
	const glm::quat& getRotation() { updateOnDirty(); return m_rotation; }
	const glm::vec3& getScale() { updateOnDirty(); return m_scale; }
	const glm::mat4& getMatrix() const { return m_transform; }

	void updateMatrix();

private:
	glm::vec3 m_position;
	glm::quat m_rotation;
	glm::vec3 m_scale;
	glm::mat4 m_transform;
	//
	void updateOnDirty();
	bool m_matrixDirty; // is dirty if matrix updated but not components
};

void TransformComponent::setPositionToMatrix(const glm::vec3& p_position)
{
	updateOnDirty();
	m_position = p_position;
	updateMatrix();
}

void TransformComponent::setRotationToMatrix(const glm::quat& p_rotation)
{
	updateOnDirty();
	m_rotation = p_rotation;
	updateMatrix();
}

void TransformComponent::setScaleToMatrix(const glm::vec3& p_scale)
{
	updateOnDirty();
	m_scale = p_scale;
	updateMatrix();
}

void TransformComponent::setMatrixToComponents(const glm::mat4& p_mat)
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
	glm::mat4 scale = glm::scale(glm::mat4(1.0f), m_scale);
	m_transform = translate * rotate * scale; // is the scale * rotate * translate
}

void TransformComponent::setMatrix(const glm::mat4& p_mat)
{
	m_transform = p_mat;
	m_matrixDirty = true;
}

void TransformComponent::updateOnDirty()
{
	if (m_matrixDirty)
	{
		setMatrixToComponents(m_transform);
		m_matrixDirty = false;
	}
}

void TransformComponent::setPosRotToMatrix(const glm::vec3& p_position, const glm::quat& p_rotation)
{
	updateOnDirty();
	m_position = p_position;
	m_rotation = p_rotation;
	updateMatrix();
}

void TransformComponent::setPosScaleToMatrix(const glm::vec3& p_position, const glm::vec3& p_scale)
{
	updateOnDirty();
	m_position = p_position;
	m_scale = p_scale;
	updateMatrix();
}

void TransformComponent::setRotScaleToMatrix(const glm::quat& p_rotation, const glm::vec3& p_scale)
{
	updateOnDirty();
	m_rotation = p_rotation;
	m_scale = p_scale;
	updateMatrix();
}

void TransformComponent::setAllToMatrix(const glm::vec3& p_position, const glm::quat& p_rotation, const glm::vec3& p_scale)
{
	updateOnDirty();
	m_position = p_position;
	m_rotation = p_rotation;
	m_scale = p_scale;
	updateMatrix();
}
