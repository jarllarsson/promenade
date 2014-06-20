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
	TransformComponent(float p_x = 0.0f, float p_y = 0.0f, float p_z = 0.0f);

	TransformComponent(glm::vec3& p_position,
		glm::quat& p_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), // note, w first
		glm::vec3& p_scale = glm::vec3(1.0f));

	TransformComponent(glm::vec3& p_position,
		glm::vec3& p_scale,
		glm::quat& p_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f)); // note, w first


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

	glm::vec3 getForward() const;
	glm::vec3 getUp() const;
	glm::vec3 getRight() const;
	glm::vec3 getDown() const;
	glm::vec3 getLeft() const;
	glm::vec3 getBackward() const;

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
