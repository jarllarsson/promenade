#pragma once
#include <glm\gtc\type_ptr.hpp>
#include <exception>

// =======================================================================================
//                                      MathHelp
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # MathHelp
/// 
/// 24-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------
#define PI 3.141592653589793238462643383279502884197169399375105820
#define HALFPI 0.5*PI
#define TWOPI 2.0*PI
#define TORAD PI/180
#define TODEG 180/PI
#define PIOVER180 TORAD

namespace MathHelp
{
	size_t roundup(int group_size, int global_size);

	void decomposeTRS(const glm::mat4& m, glm::vec3& scaling,
		glm::mat4& rotation, glm::vec3& translation);

	glm::vec3 transformDirection(const glm::mat4& m, glm::vec3& p_dir);
	glm::vec3 transformPosition(const glm::mat4& m, glm::vec3& p_pos);

	glm::vec3 toVec3(const glm::vec4& p_v);

	class CMatrix
	{
	public:
		float** m;
		unsigned int m_cols;
		unsigned int m_rows;

		CMatrix()
		{
			m_cols = 4; m_rows = 4;
			init();
		}

		CMatrix(unsigned int p_rows, unsigned int p_cols)
		{
			m_rows = p_rows; m_cols = p_cols;
			init();
		}

		~CMatrix()
		{
			for (int i = 0; i < m_rows; i++)
			{
				delete[] m[i];
			}
			delete[] m;
		}

		int test();


		float& operator() (unsigned int p_row, unsigned int p_column) const
		{
			return m[p_row][p_column];
		}


		static CMatrix mul(const CMatrix& p_ma, const CMatrix& p_mb)
		{
			if (p_ma.m_cols != p_mb.m_rows)
			{
				throw std::exception("Input matrices does not have matching column- and row sizes CMatrix::mul");
				return CMatrix();
			}
			CMatrix res(p_ma.m_rows, p_mb.m_cols);
			int y = p_ma.m_cols;
			for (unsigned int i = 0; i < res.m_rows; i++)
				for (unsigned int j = 0; j < res.m_cols; j++)
				{
				float s = 0.0f;
				for (unsigned int x = 0; x < y; x++)
				{
					s += p_ma(i,x) * p_mb(x,j);
				}
				// Debug.Log(i+","+j+"="+s);
				res(i, j) = s;
				}
			return res;
		}

		static CMatrix transpose(const CMatrix& p_m)
		{
			CMatrix res(p_m.m_cols, p_m.m_rows);
			for (int i = 0; i < p_m.m_rows; i++)
				for (int j = 0; j < p_m.m_cols; j++)
				{
				res(j, i) = p_m(i, j);
				}
			return res;
		}

		static CMatrix operator *(const CMatrix& p_ma, const CMatrix& p_mb)
		{
			return mul(p_ma, p_mb);
		}

		static CMatrix operator *(float p_s, const CMatrix& p_m)
		{
			CMatrix res(p_m.m_rows, p_m.m_cols);
			for (int i = 0; i < p_m.m_rows; i++)
				for (int j = 0; j < p_m.m_cols; j++)
				{
				res(i, j) = p_s*p_m(i, j);
				}
			return res;
		}

		static float dot(const CMatrix& p_ma, const CMatrix& p_mb)
		{
			if (p_ma.m_rows != p_mb.m_rows ||
				p_ma.m_cols != p_mb.m_cols) 
			{
				throw std::exception("Input matrices not of same size in CMatrix::dot");
				return -1.0f;
			};
			float sum = 0.0f;
			for (int i = 0; i < p_ma.m_rows; i++)
				for (int j = 0; j < p_mb.m_cols; j++)
				{
				sum += p_ma(i, j) * p_mb(i, j);
				}
			return sum;
		}
	private:
			void init()
			{
				m = new float*[m_rows];
				for (int i = 0; i < m_rows; i++)
				{
					m[i] = new float[m_cols];
					for (int j = 0; j = m_cols; j++)
						m[i][j] = 0.0f;
				}
			}
	};
	
};