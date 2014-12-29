#pragma once
#include <windows.h>
// =======================================================================================
//                                      Time
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Timer class
///        
/// # Time
/// 
/// 21-6-2014 Jarl Larsson
///---------------------------------------------------------------------------------------

class Time
{
public:
	Time() {}
	virtual ~Time() {}

	static LARGE_INTEGER getTimeStamp();
	static LARGE_INTEGER getTicksPerSecond();
	static double		 getSecondsPerTick();
	static double		 getTimeSeconds();
protected:

private:
	static LARGE_INTEGER initFrequency();
	static double initSecPerTick();
	static LARGE_INTEGER m_ticksPerSec;
	static double		 m_secondsPerTick;
};