#pragma once

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

class PieceWiseLinear
{
public:
	enum InitType
	{
		SIN,
		COS,
		COS_INV_NORM, // inverted and normalized cos
		COS_INV_NORM_PADDED, // same as above but with padding at end
		HALF_SIN,      // half sine
		FLAT,        // flat zero
		LIN_INC,    // Linear increase
		RND,
		LIN_DEC
	};


	PieceWiseLinear();
	PieceWiseLinear(InitType p_initFunction);
	~PieceWiseLinear();

	unsigned int getSize() const;
	float getNormalizedIdx(unsigned int p_idx) const;
	float lerpGet(float p_phi) const;

protected:
private:
	void init();
	void clear();

	// Number of data points, locked for now
	static const unsigned int c_size = 4;

	// The data
	float* m_dataPoints;
};