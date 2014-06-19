#pragma once
#include <ColorPalettes.h>
#include <glm\gtc\type_ptr.hpp>
// =======================================================================================
//                                      InstanceData
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # InstanceData
/// 
/// 24-3-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

struct InstanceDataTransform
{
public:
	glm::mat4 m_transform;
};


struct InstanceDataTransformColor
{
public:
	glm::mat4 m_transform;
	Color4f	  m_color;
};