#include "ToString.h"
#include <glm\gtc\type_ptr.hpp>
#include <string>
#include <sstream>



template <>
std::string ToString<glm::vec3>(const glm::vec3& val)
{
	std::string vecstr = ToString(val.x) + std::string(",") + ToString(val.y) + std::string(",") + ToString(val.z);
	return vecstr;
}

template <>
std::string ToString<glm::vec4>(const glm::vec4& val)
{
	std::string vecstr = ToString(val.x) + std::string(",") + ToString(val.y) + std::string(",") + ToString(val.z) + std::string(",") + ToString(val.w);
	return vecstr;
}

template <>
std::string ToString<glm::mat4>(const glm::mat4& val)
{
	std::string matstr = "\nmat4=";
	for (unsigned int f = 0; f < (unsigned int)val.length(); f++)
		matstr += std::string("\nm[") + ToString(f) + std::string("] = ") + ToString(val[f]);
	return matstr;
}
// 
// template <>
// std::string ToString<std::vector<T>>(const std::vector<T>& val)
// {
// 	std::string liststr = "\nlist=";
// 	for (unsigned int f = 0; f < (unsigned int)val.size(); f++)
//		liststr += std::string("\n[") + ToString(f) + std::string("] = ") + ToString(val[f]);
// 	return liststr;
// }
