#pragma once

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

class RenderComponent
{
public:
	RenderComponent(/*unsigned int p_meshIdx, unsigned int p_instanceListIdx, */unsigned int p_instanceIdx)
	{
// 		m_meshIdx = p_meshIdx;
// 		m_instanceListIdx = p_instanceListIdx;
		m_instanceIdx = p_instanceIdx;
	};

// 	unsigned int getMeshIdx();
// 	unsigned int getInstanceListIdx();
	unsigned int getInstanceIdx();

private:
	unsigned int /*m_meshIdx, m_instanceListIdx, */m_instanceIdx;
};

// unsigned int RenderComponent::getMeshIdx()
// {
// 	return m_meshIdx;
// }
// 
// unsigned int RenderComponent::getInstanceListIdx()
// {
// 	return m_instanceListIdx;
// }

unsigned int RenderComponent::getInstanceIdx()
{
	return m_instanceIdx;
}
