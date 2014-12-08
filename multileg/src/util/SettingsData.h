#pragma once
#include <string>

// =======================================================================================
//                                      SettingsData
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Class reflecting a settings file
///        
/// # SettingsData
/// Detailed description.....
/// Created on: 26-11-2014 
///---------------------------------------------------------------------------------------

class SettingsData
{
public:
	SettingsData();
	~SettingsData();

	bool m_fullscreen;

	std::string m_appMode;

	int m_wwidth;

	int m_wheight;

	std::string m_simMode;

	std::string m_pod;

	std::string m_execMode;

	int m_charcount_serial;

	int m_parallel_invocs;

	float m_charOffsetX;

	bool m_startPaused;

	int m_optimizationSteps;

	float m_optW_fd, m_optW_fv, m_optW_fh, m_optW_fr, m_optW_fp;

protected:
private:
};