#pragma once
#include <string>
#include <wtypes.h>

// =======================================================================================
//                                      StrTools
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # StrTools
/// Detailed description.....
/// Created on: 27-11-2014 
///---------------------------------------------------------------------------------------

std::wstring stringToWstring(const std::string& s);

LPCWSTR stringToLPCWSTR(const std::string& s);

int stringToInt(const std::string& s);