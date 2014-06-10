#pragma once
#include <exception>

// =======================================================================================
//                                      CMatrix
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Matrix of variable dimensions
///        
/// # CMatrix
/// 
/// 10-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class CMatrix
{
public:
	float** m;
	unsigned int m_cols;
	unsigned int m_rows;

	CMatrix();

	CMatrix(unsigned int p_rows, unsigned int p_cols);

	~CMatrix();


	float& operator() (unsigned int p_row, unsigned int p_column) const;


	static CMatrix mul(const CMatrix& p_ma, const CMatrix& p_mb);

	static CMatrix transpose(const CMatrix& p_m);

	static CMatrix operator *(const CMatrix& p_ma, const CMatrix& p_mb);

	static CMatrix operator *(float p_s, const CMatrix& p_m);

	static float dot(const CMatrix& p_ma, const CMatrix& p_mb);
private:
	void init();
};