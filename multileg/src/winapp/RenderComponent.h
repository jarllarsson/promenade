#pragma once
#include <Artemis.h>

// =======================================================================================
//                                      RenderComponent
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Tells that an entity should be rendered
///        
/// # RenderComponent
/// 
/// 16-5-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class RenderComponent : public artemis::Component
{
public:
	RenderComponent(/*unsigned int p_meshIdx, unsigned int p_instanceListIdx, unsigned int p_instanceIdx*/)
	{
// 		m_meshIdx = p_meshIdx;
// 		m_instanceListIdx = p_instanceListIdx;
		m_instanceIdx = -1;
	};

// 	unsigned int getMeshIdx();
// 	unsigned int getInstanceListIdx();
	int getInstanceIdx();
	void setInstanceIdx(unsigned int p_idx)
	{
		m_instanceIdx = (int)p_idx;
	}
private:
	int /*m_meshIdx, m_instanceListIdx, */m_instanceIdx;
};
