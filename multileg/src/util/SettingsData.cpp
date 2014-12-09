#include "SettingsData.h"

SettingsData::SettingsData()
{
	m_fullscreen = false;

	m_appMode = "graphics";

	m_wwidth = 600;

	m_wheight = 800;

	m_simMode = "run";

	m_measurementRuns = 1;

	m_pod = "biped";

	m_execMode = "serial";

	m_charcount_serial = 1;

	m_parallel_invocs = 1;

	m_charOffsetX = 0.0f;

	m_startPaused = false;

	m_optmesSteps = 800;

	m_optW_fd=0.0f;
	m_optW_fv=0.0f;
	m_optW_fh=0.0f;
	m_optW_fr=0.0f;
	m_optW_fp=0.0f;
}

SettingsData::~SettingsData()
{

}
