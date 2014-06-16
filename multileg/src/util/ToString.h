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
#include <glm\gtc\type_ptr.hpp>
#include <string>
#include <sstream>
#include <vector>

template <typename T>
std::string ToString(const T& val)
{
	// Convert input value to string
	// using stringstream
	std::stringstream strStream;
	strStream << val;
	return strStream.str();
}

template <>
std::string ToString<glm::vec3>(const glm::vec3& val);

template <>
std::string ToString<glm::vec4>(const glm::vec4& val);

template <>
std::string ToString<glm::mat4>(const glm::mat4& val);


template <typename T>
std::string ToString(const std::vector<T>& val)
{
	std::string liststr = "\nlist=";
	for (unsigned int f = 0; f < (unsigned int)val.size(); f++)
		liststr += std::string("\n[") + ToString(f) + std::string("] = ") + ToString(val[f]);
	return liststr;
}



