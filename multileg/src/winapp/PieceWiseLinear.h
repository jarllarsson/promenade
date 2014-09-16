#pragma once
#include <vector>
#include <IOptimizable.h>

// =======================================================================================
//                                      PieceWiseLinear
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Piece wise linear function, a function defined by data points 
///			and accessed by linear interpolation.
///        
/// # PieceWiseLinear
/// 
/// 17-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class PieceWiseLinear : public IOptimizable
{
public:
	enum InitType
	{
		SIN,
		COS,
		COS_INV_NORM, // inverted and normalized cos
		COS_INV_NORM_PADDED, // same as above but with padding at end
		HALF_SIN,      // half sine
		FLAT,        // flat 0
		HALF,		// flat 0.5
		FULL,		// flat 1
		LIN_INC,    // Linear increase
		LIN_DEC
	};


	PieceWiseLinear();
	PieceWiseLinear(InitType p_initFunction);
	PieceWiseLinear(const PieceWiseLinear& p_copy);
	PieceWiseLinear& operator = (const PieceWiseLinear& p_rhs);
	~PieceWiseLinear();

	void reset(InitType p_initFunction=InitType::FLAT, float p_scale=1.0f);
	unsigned int getSize() const;
	float getNormalizedIdx(unsigned int p_idx) const;
	float lerpGet(float p_phi) const;
	float get(unsigned int p_idx) const;

	// Optimization
	virtual std::vector<float> getParams();
	virtual void consumeParams(std::vector<float>& p_other);
	virtual std::vector<float> getParamsMax();
	virtual std::vector<float> getParamsMin();

protected:
	// The data
	float* m_dataPoints;
private:
	void init();
	void init(const PieceWiseLinear& p_copy);
	void clear();

	// Number of data points, locked for now
	static const unsigned int c_size = 4;
	float m_scale; // used for optimization
};