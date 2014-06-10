#pragma once
#include <CMatrix.h>

TEST_CASE("CMatrix creation", "[CMatrix,creation]") {
	// Test default constructor
	//REQUIRE(CMatrix(4, 4) == CMatrix());
	// Test member assignment
	CMatrix mat(100, 10);
	REQUIRE(mat.m_rows == 100);
	REQUIRE(mat.m_cols == 10);
}

TEST_CASE("CMatrix member assignment", "[CMatrix,assignment]") {
	// Test member assignment
	CMatrix mat(100, 10);
	mat(50, 4) = 4.0f;
	REQUIRE(mat(50, 4) == 4.0f);
	//
	for (int x = 0; x < mat.m_rows; x++)
		for (int y = 0; y < mat.m_cols; y++)
		{
		mat(x, y) = (x + y - 10.0f)*(x + y);
		}
	for (int x = 0; x < mat.m_rows; x++)
		for (int y = 0; y < mat.m_cols; y++)
		{
		REQUIRE(mat(x, y) == Approx((x + y - 10.0f)*(x + y)));
		}
	for (int y = 0; y < mat.m_cols; y++)
		for (int x = 0; x < mat.m_rows; x++)
		{
		REQUIRE(mat(x, y) == Approx((x + y - 10.0f)*(x + y)));
		}
}

TEST_CASE("CMatrix transpose", "[CMatrix,transpose]") {
	// Test member assignment
	CMatrix mat(100, 10);
	//
	for (int x = 0; x < mat.m_rows; x++)
	for (int y = 0; y < mat.m_cols; y++)
	{
	mat(x, y) = (x + y - 10.0f)*(x + y);
	}
	CMatrix matT = CMatrix::transpose(mat);
	REQUIRE(matT.m_cols == mat.m_rows);
	REQUIRE(matT.m_rows == mat.m_cols);
	for (int x = 0; x < mat.m_rows; x++)
	for (int y = 0; y < mat.m_cols; y++)
	{
	REQUIRE(matT(y, x) == Approx((x + y - 10.0f)*(x + y)));
	}
}

TEST_CASE("CMatrix multiply", "[CMatrix,multiply]") {
	// Test member assignment
	CMatrix mat1(20, 10);
	CMatrix mat2(10, 20);
	//
	for (int x = 0; x < mat1.m_rows; x++)
	for (int y = 0; y < mat1.m_cols; y++)
	{
	mat1(x, y) = (float)x+(float)y*(float)(x+y);
	}
	for (int x = 0; x < mat2.m_rows; x++)
	for (int y = 0; y < mat2.m_cols; y++)
	{
	mat2(x, y) = (float)(y+x)+((float)x*2.0f)-((float)y*0.5f);
	}
	CMatrix matR = mat1*mat2;
	int y = mat1.m_cols;
	REQUIRE(matR.m_rows == mat1.m_rows);
	REQUIRE(matR.m_cols == mat2.m_cols);
	for (unsigned int i = 0; i < mat1.m_rows; i++)
	for (unsigned int j = 0; j < mat2.m_cols; j++)
	{
		float s = 0.0f;
		for (unsigned int x = 0; x < y; x++)
		{
			s += mat1(i, x) * mat2(x, j);
		}
		REQUIRE(matR(i, j) == Approx(s));
	}
}