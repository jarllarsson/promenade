// =======================================================================================
//                                      ShaderStageConfig
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Contains information for a specific shader stage.
///        
/// # ShaderEntryProfile
/// Detailed description.....
/// Created on: 3-12-2012 
/// Tweaked: 19-4-2013 Jarl Larsson
/// Based on ShaderFactory code of Amalgamation 
///---------------------------------------------------------------------------------------
#pragma once
#include <string>
#include <wtypes.h>

using namespace std;

struct ShaderStageConfig
{
	ShaderStageConfig(const string& p_filePath, const string& p_entryPoint, const string& p_version)
	{
		filePath = p_filePath;
		entryPoint = p_entryPoint;
		version = p_version;
	}

	//ShaderStageConfig(const string& p_filePath, const string& p_entryPoint)
	//{
	//	filePath = p_filePath;
	//	entryPoint = p_entryPoint;
	//	version = "";
	//}
	//
	////ShaderStageConfig(const string& p_entryPoint, const string& p_version)
	////{
	////	filePath="";
	////	entryPoint = p_entryPoint;
	////	version = p_version;
	////}
	//
	//ShaderStageConfig(const string& p_entryPoint)
	//{
	//	filePath="";
	//	entryPoint = p_entryPoint;
	//	version="";
	//}



	string filePath;
	string entryPoint;
	string version;
};