#include "CMatrix.h"


CMatrix::CMatrix()
{
	m = NULL;
	m_cols = 4; m_rows = 4;
	init();
}

CMatrix::CMatrix(unsigned int p_rows, unsigned int p_cols)
{
	m = NULL;
	m_rows = p_rows; m_cols = p_cols;
	init();
}

CMatrix::CMatrix(const CMatrix& p_copy)
{
	m = NULL;
	m_rows = p_copy.m_rows; m_cols = p_copy.m_cols;
	init(p_copy);
}

CMatrix::~CMatrix()
{
	clear();
}



float& CMatrix::operator() (unsigned int p_row, unsigned int p_column) const
{
	return m[p_row][p_column];
}


CMatrix CMatrix::mul(const CMatrix& p_ma, const CMatrix& p_mb)
{
	if (p_ma.m_cols != p_mb.m_rows)
	{
		throw std::exception("Input matrices does not have matching column- and row sizes CMatrix::mul");
		return CMatrix();
	}
	CMatrix res(p_ma.m_rows, p_mb.m_cols);
	unsigned int y = p_ma.m_cols;
	for (unsigned int i = 0; i < res.m_rows; i++)
	for (unsigned int j = 0; j < res.m_cols; j++)
	{
		float s = 0.0f;
		for (unsigned int x = 0; x < y; x++)
		{
			s += p_ma(i, x) * p_mb(x, j);
		}
		res(i, j) = s;
	}
	return res;
}

CMatrix CMatrix::transpose(const CMatrix& p_m)
{
	CMatrix res(p_m.m_cols, p_m.m_rows);
	for (unsigned int i = 0; i < res.m_rows; i++)
	for (unsigned int j = 0; j < res.m_cols; j++)
	{
		res(i, j) = p_m(j, i);
	}
	return res;
}

CMatrix CMatrix::operator *(const CMatrix& p_mb) const
{
	return mul(*this, p_mb);
}

CMatrix CMatrix::operator *(float p_s) const
{
	CMatrix res(m_rows, m_cols);
	for (unsigned int i = 0; i < m_rows; i++)
	for (unsigned int j = 0; j < m_cols; j++)
	{
		res(i, j) = p_s*m[i][j];
	}
	return res;
}


bool CMatrix::operator==(const CMatrix& p_mb) const
{
	if (m_rows != p_mb.m_rows ||
		m_cols != p_mb.m_cols)
		return false;
	for (unsigned int i = 0; i < m_rows; i++)
	for (unsigned int j = 0; j < m_cols; j++)
	{
		if (m[i][j] != p_mb(i, j)) return false;
	}
	return true;
}

float CMatrix::dot(const CMatrix& p_ma, const CMatrix& p_mb)
{
	if (p_ma.m_rows != p_mb.m_rows ||
		p_ma.m_cols != p_mb.m_cols)
	{
		throw std::exception("Input matrices not of same size in CMatrix::dot");
		return -1.0;
	};
	float sum = 0.0;
	for (unsigned int i = 0; i < p_ma.m_rows; i++)
	for (unsigned int j = 0; j < p_mb.m_cols; j++)
	{
	sum += p_ma(i, j) * p_mb(i, j);
	}
	return sum;
}

void CMatrix::init()
{
	m = new float*[m_rows];
	for (unsigned int i = 0; i < m_rows; i++)
	{
		m[i] = new float[m_cols];
		for (unsigned int j = 0; j < m_cols; j++)
			m[i][j] = 0.0f;
	}
}

void CMatrix::init(const CMatrix& p_copy)
{
	m = new float*[m_rows];
	for (unsigned int i = 0; i < m_rows; i++)
	{
		m[i] = new float[m_cols];
		for (unsigned int j = 0; j < m_cols; j++)
			m[i][j] = p_copy(i,j);
	}
}

CMatrix& CMatrix::operator=(const CMatrix& p_mb)
{
	if (this != &p_mb) {
		clear();
		m_rows = p_mb.m_rows;
		m_cols = p_mb.m_cols;
		init(p_mb);
	}
	return *this;
}

void CMatrix::clear()
{
	for (unsigned int i = 0; i < m_rows; i++)
	{
		delete[] m[i];
	}
	delete[] m;
}
