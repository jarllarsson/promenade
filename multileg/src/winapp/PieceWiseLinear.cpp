#include "PieceWiseLinear.h"

PieceWiseLinear::PieceWiseLinear()
{
	init();
}

PieceWiseLinear::PieceWiseLinear(InitType p_initFunction)
{

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
	for (int i = 0; i < getSize(); i++)
	{
		m_dataPoints[i] = 0.0f;
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
	//Debug.Log(realTime + ": " + lowIdx + "->" + hiIdx + " [t" + lin + "]");
	//Debug.Log(hi);
	float val = 0.0f;
	val = (m_tuneDataPoints[lowIdx], m_tuneDataPoints[hiIdx], lin);
	return val;
}
