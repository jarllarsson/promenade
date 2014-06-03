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
		m_matrixDirty = false;
		m_transformRenderDirty = true;
		m_position = glm::vec3(p_x,p_y,p_z);
		m_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
		m_scale = glm::vec3(1.0f);
		updateMatrix();

	};

	TransformComponent(glm::vec3& p_position, 
					   glm::quat& p_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), // note, w first
					   glm::vec3& p_scale  = glm::vec3(1.0f))
	{
		m_matrixDirty = false;
		m_transformRenderDirty = true;
		m_position = p_position;
		m_rotation = p_rotation;
		m_scale = p_scale;
		updateMatrix();
	};

	TransformComponent(glm::vec3& p_position,
					   glm::vec3& p_scale,
					   glm::quat& p_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) // note, w first
	{
		m_matrixDirty = false;
		m_transformRenderDirty = true;
		m_position = p_position;
		m_rotation = p_rotation;
		m_scale = p_scale;
		updateMatrix();
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

	const glm::vec3& getPosition() { updateComponentsOnMatrixDirty(); return m_position; }
	const glm::quat& getRotation() { updateComponentsOnMatrixDirty(); return m_rotation; }
	const glm::vec3& getScale() { updateComponentsOnMatrixDirty(); return m_scale; }
	const glm::mat4& getMatrix() const { return m_transform; }
	const glm::mat4* getMatrixPtr() const { return &m_transform; }
	const glm::mat4 getMatrixPosRot() const;

	void updateMatrix();

	bool isTransformRenderDirty();
	void unsetTransformRenderDirty();

private:
	glm::vec3 m_position;
	glm::quat m_rotation;
	glm::vec3 m_scale;
	glm::mat4 m_transform;
	//
	void updateComponentsOnMatrixDirty();
	bool m_matrixDirty; // is dirty if matrix updated but not components
	bool m_transformRenderDirty; // any change
};

void TransformComponent::setPositionToMatrix(const glm::vec3& p_position)
{
	updateComponentsOnMatrixDirty();
	m_position = p_position;
	updateMatrix();
	m_transformRenderDirty = true;
}

void TransformComponent::setRotationToMatrix(const glm::quat& p_rotation)
{
	updateComponentsOnMatrixDirty();
	m_rotation = p_rotation;
	updateMatrix();
	m_transformRenderDirty = true;
}

void TransformComponent::setScaleToMatrix(const glm::vec3& p_scale)
{
	updateComponentsOnMatrixDirty();
	m_scale = p_scale;
	updateMatrix();
	m_transformRenderDirty = true;
}

void TransformComponent::setMatrixToComponents(const glm::mat4& p_mat)
{
	m_transform = p_mat;
	glm::mat4 rotation;
	// decompose the matrix to its components
	MathHelp::decomposeTRS(m_transform, m_scale, rotation, m_position);
	// get rotation quat
	m_rotation = glm::quat(rotation);
	m_transformRenderDirty = true;
}

// Update transform from component values
void TransformComponent::updateMatrix()
{
	glm::mat4 translate = glm::translate(glm::mat4(1.0f), m_position);
	glm::mat4 rotate = glm::mat4_cast(m_rotation);
// 	glm::vec3 test = glm::eulerAngles(m_rotation);
// 	glm::quat castBack = glm::quat(rotate);
// 	glm::vec3 test2 = glm::eulerAngles(castBack); // are they the same?
	glm::mat4 scale = glm::scale(glm::mat4(1.0f), m_scale);
	m_transform = translate * rotate * scale; // is the scale * rotate * translate
	m_transformRenderDirty = true;
}

void TransformComponent::setMatrix(const glm::mat4& p_mat)
{
	m_transform = p_mat;
	m_matrixDirty = true;
	m_transformRenderDirty = true;
}

void TransformComponent::updateComponentsOnMatrixDirty()
{
	if (m_matrixDirty)
	{
		setMatrixToComponents(m_transform);
		m_matrixDirty = false;
		m_transformRenderDirty = true;
	}
}

void TransformComponent::setPosRotToMatrix(const glm::vec3& p_position, const glm::quat& p_rotation)
{
	updateComponentsOnMatrixDirty();
	m_position = p_position;
	m_rotation = p_rotation;
	updateMatrix();
	m_transformRenderDirty = true;
}

void TransformComponent::setPosScaleToMatrix(const glm::vec3& p_position, const glm::vec3& p_scale)
{
	updateComponentsOnMatrixDirty();
	m_position = p_position;
	m_scale = p_scale;
	updateMatrix();
	m_transformRenderDirty = true;
}

void TransformComponent::setRotScaleToMatrix(const glm::quat& p_rotation, const glm::vec3& p_scale)
{
	updateComponentsOnMatrixDirty();
	m_rotation = p_rotation;
	m_scale = p_scale;
	updateMatrix();
	m_transformRenderDirty = true;
}

void TransformComponent::setAllToMatrix(const glm::vec3& p_position, const glm::quat& p_rotation, const glm::vec3& p_scale)
{
	updateComponentsOnMatrixDirty();
	m_position = p_position;
	m_rotation = p_rotation;
	m_scale = p_scale;
	updateMatrix();
	m_transformRenderDirty = true;
}

bool TransformComponent::isTransformRenderDirty()
{
	return m_transformRenderDirty;
}

void TransformComponent::unsetTransformRenderDirty()
{
	m_transformRenderDirty = false;
}

const glm::mat4 TransformComponent::getMatrixPosRot() const
{
	glm::mat4 translate = glm::translate(glm::mat4(1.0f), m_position);
	glm::mat4 rotate = glm::mat4_cast(m_rotation);
	return translate * rotate; // is the scale * rotate * translate
}
