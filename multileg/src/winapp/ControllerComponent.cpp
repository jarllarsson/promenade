#include "ControllerComponent.h"

std::vector<float> ControllerComponent::getParams()
{
	std::vector<float> params;
	params.push_back(m_player.getParams());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		params.push_back(m_legFrames[i].getParams());
	}
}


void ControllerComponent::consumeParams(const std::vector<float>& p_other)
{
	m_player.consumeParams(p_other);
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		m_legFrames[i].consumeParams();
	}
}

std::vector<float> ControllerComponent::getParamsMax()
{
	std::vector<float> paramsmax;
	paramsmax.push_back(m_player.getParamsMax());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		paramsmax.push_back(m_legFrames[i].getParamsMax());
	}
}

std::vector<float> ControllerComponent::getParamsMin()
{
	std::vector<float> paramsmin;
	paramsmin.push_back(m_player.getParamsMin());
	for (int i = 0; i < m_legFrames.size(); i++)
	{
		paramsmin.push_back(m_legFrames[i].getParamsMin());
	}
}
