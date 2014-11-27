#include <vector>
#include "CurrentPathHelper.h"
#include <windows.h>
#include "DebugPrint.h"

std::string GetExecutablePathWithName()
{
	std::vector<char>executablePath(MAX_PATH);

	// Try to get the executable path with a buffer of MAX_PATH characters.
	DWORD result = ::GetModuleFileName(
		nullptr, &executablePath[0], static_cast<DWORD>(executablePath.size())
		);

	// As long the function returns the buffer size, it is indicating that the buffer
	// was too small. Keep enlarging the buffer by a factor of 2 until it fits.
	while (result == executablePath.size()) {
		executablePath.resize(executablePath.size() * 2);
		result = ::GetModuleFileName(
			nullptr, &executablePath[0], static_cast<DWORD>(executablePath.size())
			);
	}

	// If the function returned 0, something went wrong
	if (result == 0) 
	{
		DEBUGWARNING(("Failed to find path of executable! Current working directory will be tried as path."));
	}

	// We've got the path, construct a standard string from it
	return std::string(executablePath.begin(), executablePath.begin() + result);
}

std::string GetExecutablePathDirectory(std::string* p_outOptionalExeFileName/*=NULL*/)
{
	std::string fullpath = GetExecutablePathWithName();
	unsigned found = fullpath.find_last_of("/\\");
	// path 
	std::string path = fullpath.substr(0, found)+"/";
	// name 
	std::string filename = fullpath.substr(found + 1);
	//
	if (p_outOptionalExeFileName != NULL)
		*p_outOptionalExeFileName = filename;
	return path;
}