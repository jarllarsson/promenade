#pragma once

// =======================================================================================
//                                      ToString
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # ToString
/// 
/// 28-11-2012 Jarl Larsson
///---------------------------------------------------------------------------------------

#define Stringify(x) #x

#include <sstream>
using namespace std;

template <class T>
std::string toString (const T& val)
{
	// Convert input value to string
	// using stringstream
	stringstream strStream;
	strStream << val;
	return strStream.str();
}