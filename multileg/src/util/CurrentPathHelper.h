#pragma once
#include <string>

// =======================================================================================
//                                      CurrentPathHelper
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Brief
///        
/// # CurrentPathHelper
/// Detailed description.....
/// Created on: 27-11-2014 
///---------------------------------------------------------------------------------------

std::string GetExecutablePathWithName();

std::string GetExecutablePathDirectory(std::string* p_outOptionalExeFileName=NULL);