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

	CMatrix(const CMatrix& p_copy);

	CMatrix(unsigned int p_rows, unsigned int p_cols);

	~CMatrix();


	float& operator() (unsigned int p_row, unsigned int p_column) const;


	static CMatrix mul(const CMatrix& p_ma, const CMatrix& p_mb);

	static CMatrix transpose(const CMatrix& p_m);

	CMatrix operator *(const CMatrix& p_ma) const;

	CMatrix operator *(float p_s) const;

	bool operator == (const CMatrix& p_mb) const;

	static float dot(const CMatrix& p_ma, const CMatrix& p_mb);
private:
	void init();
	void init(const CMatrix& p_copy);
};