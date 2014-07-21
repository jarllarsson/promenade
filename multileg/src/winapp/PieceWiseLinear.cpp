#include "PieceWiseLinear.h"
#include <MathHelp.h>
#include <OptimizableHelper.h>
#include <DebugPrint.h>

PieceWiseLinear::PieceWiseLinear()
{
	init();
}

PieceWiseLinear::PieceWiseLinear(InitType p_initFunction)
{
	init();
	reset(p_initFunction);
}

PieceWiseLinear::PieceWiseLinear(const PieceWiseLinear& p_copy)
{
	init(p_copy);
}

PieceWiseLinear& PieceWiseLinear::operator=(const PieceWiseLinear& p_rhs)
{
	if (this != &p_rhs) 
	{
		clear();
		init(p_rhs);
	}
	return *this;
}

PieceWiseLinear::~PieceWiseLinear()
{
	clear();
}

unsigned int PieceWiseLinear::getSize() const
{
	return c_size;
}

void PieceWiseLinear::init()
{
	m_dataPoints = new float[c_size];
	for (unsigned int i = 0; i < getSize(); i++)
	{
		m_dataPoints[i] = 0.0f;
	}
}

void PieceWiseLinear::init(const PieceWiseLinear& p_copy)
{
	m_dataPoints = new float[c_size];
	for (unsigned int i = 0; i < getSize(); i++)
	{
		m_dataPoints[i] = p_copy.get(i);
	}
}

void PieceWiseLinear::clear()
{
	delete[] m_dataPoints;
}

float PieceWiseLinear::getNormalizedIdx(unsigned int p_idx) const
{
	return (float)p_idx / (float)((int)getSize() - 1);
}

float PieceWiseLinear::lerpGet(float p_phi) const
{
	int signedSz = static_cast<int>(getSize());
	float scaledPhi = (float)(signedSz - 1) * p_phi;
	// lower bound idx (never greater than last idx)
	int lowIdx = (int)(scaledPhi) % (signedSz);
	// higher bound idx (loops back to 1 if over)
	int hiIdx = ((int)(scaledPhi)+1) % (signedSz);
	// get amount of interpolation by subtracting the base from current
	float lin = p_phi * (float)(signedSz - 1) - (float)lowIdx;
	float val = MathHelp::flerp(m_dataPoints[lowIdx], m_dataPoints[hiIdx], lin);
	return val;
}

void PieceWiseLinear::reset(InitType p_initFunction/*=InitType::FLAT*/, float p_scale/*=1.0f*/)
{
	for (unsigned int i = 0; i < getSize(); i++)
	{
		float t = getNormalizedIdx(i);
		switch (p_initFunction)
		{
		case InitType::SIN:
			m_dataPoints[i] = p_scale*sin(t * 2.0f * (float)PI);
			break;
		case InitType::COS:
			m_dataPoints[i] = p_scale * cos(t * 2.0f * (float)PI);
			break;
		case InitType::COS_INV_NORM:
			m_dataPoints[i] = p_scale * ((cos(t * 2.0f * (float)PI) - 1.0f) * -0.5f);
			break;
		case InitType::COS_INV_NORM_PADDED:
			m_dataPoints[i] = p_scale * ((cos(t * 2.0f * (float)PI) - 1.0f) * -0.5f);
			if (i > getSize() - 3 && i >= 0) m_dataPoints[i] = 0.0f;
			break;
		case InitType::HALF_SIN:
			m_dataPoints[i] = p_scale * sin(t * (float)PI);
			break;
		case InitType::FLAT:
			m_dataPoints[i] = 0.0f;
			break;
		case InitType::HALF:
			m_dataPoints[i] = p_scale * 0.5f;
			break;
		case InitType::FULL:
			m_dataPoints[i] = p_scale * 1.0f;
			break;
		case InitType::LIN_INC:
			m_dataPoints[i] = p_scale * t;
			break;
		case InitType::LIN_DEC:
			m_dataPoints[i] = p_scale * (1.0f - t);
			break;
		default:
			m_dataPoints[i] = p_scale * 0.5f;
			break;
		}
	}
}

float PieceWiseLinear::get(unsigned int p_idx) const
{
	return m_dataPoints[p_idx];
}

std::vector<float> PieceWiseLinear::getParams()
{
	DEBUGPRINT(("PIECEWISE LINEAR GETPARAMS\n"));
	std::vector<float> params;
	for (int i = 0; i < getSize(); i++)
		params.push_back(get(i));	//
	return params;
}

void PieceWiseLinear::consumeParams(std::vector<float>& p_other)
{
	for (int i = 0; i < getSize(); i++)
		OptimizableHelper::ConsumeParamsTo(p_other, &m_dataPoints[i]);
}

std::vector<float> PieceWiseLinear::getParamsMax()
{
	std::vector<float> paramsmax;
	for (int i = 0; i < getSize(); i++)
		paramsmax.push_back(100.0f);
	return paramsmax;
}

std::vector<float> PieceWiseLinear::getParamsMin()
{
	std::vector<float> paramsmin;
	for (int i = 0; i < getSize(); i++)
		paramsmin.push_back(-100.0f);
	return paramsmin;
}
