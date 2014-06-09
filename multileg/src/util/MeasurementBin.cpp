#include "MeasurementBin.h"



template<>
float MeasurementBin<float>::calculateMean()
{
	double accumulate = 0.0;
	unsigned int count = (unsigned int)m_measurements.size();
	for (unsigned int i = 0; i < count; i++)
	{
		accumulate += (double)m_measurements[i];
	}
	m_mean = accumulate / (double)count;
	return (float)m_mean;
}



template<>
float MeasurementBin<float>::calculateSTD()
{
	float mean = calculateMean();
	unsigned int count = (unsigned int)m_measurements.size();
	double squaredDistsToMean = 0.0;
	for (unsigned int i = 0; i < count; i++)
	{
		double dist = (double)m_measurements[i] - (double)mean;
		squaredDistsToMean += dist*dist;
	}
	double standardDeviation = sqrt(squaredDistsToMean / (double)count);
	m_std = standardDeviation;
	return (float)standardDeviation;
}
