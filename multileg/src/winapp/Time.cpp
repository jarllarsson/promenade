#include "Time.h"

LARGE_INTEGER Time::m_ticksPerSec = Time::initFrequency();
double Time::m_secondsPerTick = Time::initSecPerTick();

LARGE_INTEGER Time::getTimeStamp()
{
	LARGE_INTEGER stamp;
	QueryPerformanceCounter(&stamp);
	return stamp;
}

LARGE_INTEGER Time::getTicksPerSecond()
{
	return m_ticksPerSec;
}

LARGE_INTEGER Time::initFrequency()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return freq;
}

double Time::initSecPerTick()
{
	return 1.0 / (double)m_ticksPerSec.QuadPart;
}

double Time::getTimeSeconds()
{
	LARGE_INTEGER stamp = getTimeStamp();
	double time = (double)stamp.QuadPart * getSecondsPerTick();
	return time;
}

double Time::getSecondsPerTick()
{
	return m_secondsPerTick;
}
